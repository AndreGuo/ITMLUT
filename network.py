import numpy as np
import torch
import torch.nn as nn
from ailut import ailut_transform


class LutNet(nn.Module):
    def __init__(self, nc=3, n_lut=5, n_dim=17, bright_t=0.55, dark_t=0.45):
        super(LutNet, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_lut = n_lut
        self.n_dim = n_dim
        self.device = device
        self.nc = nc

        # 1/8 downscale the input img to get global prior
        self.prior_gen = nn.AvgPool2d(8)

        # soft mask to split 3 part
        self.bright_prob = BrightProb(threshold=bright_t)
        self.mid_prob = MidProb(threshold_b=bright_t, threshold_d=dark_t)
        self.dark_prob = DarkProb(threshold=dark_t)

        # LUT gen. and fusion for bright part
        self.encoder_img_b = GlobPriorEncoder(in_nc=nc, out_nc=128)
        self.lut_gen_b = LUTGenAndFuse(in_fea_nc=128, nc=nc, n_dim=n_dim, n_lut=n_lut, device=device)

        # LUT gen. and fusion for middle part
        self.encoder_img_m = GlobPriorEncoder(in_nc=nc, out_nc=128)
        self.lut_gen_m = LUTGenAndFuse(in_fea_nc=128, nc=nc, n_dim=n_dim, n_lut=n_lut, device=device)

        # LUT gen. and fusion for dark part
        self.encoder_img_d = GlobPriorEncoder(in_nc=nc, out_nc=128)
        self.lut_gen_d = LUTGenAndFuse(in_fea_nc=128, nc=nc, n_dim=n_dim, n_lut=n_lut, device=device)

    # this attribute will not be called when testing
    def init_weights(self):
        def universal_init(m):
            name = m.__class__.__name__
            if 'Conv' in name:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in name:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(universal_init)
        self.lut_gen_b.init_weights()
        self.lut_gen_m.init_weights()
        self.lut_gen_d.init_weights()

    def forward(self, img):

        prior = self.prior_gen(img)

        weights_b = self.encoder_img_b(prior)
        weights_m = self.encoder_img_m(prior)
        weights_d = self.encoder_img_d(prior)

        LUTs_b = self.lut_gen_b(weights_b)  # shape: [b, nc, n_dim, n_dim, n_dim]
        LUTs_m = self.lut_gen_m(weights_m)
        LUTs_d = self.lut_gen_b(weights_d)

        u_vertices = torch.linspace(0, 1, self.n_dim). \
            to(self.device).unsqueeze(0).unsqueeze(0).repeat(img.shape[0], 1, 1)  # shape: [b, 1, n_dim]

        avg_r = torch.mean(img[:, 0, :, :])
        avg_g = torch.mean(img[:, 1, :, :])
        avg_b = torch.mean(img[:, 2, :, :])

        exp_r = 1.4 + 0.8 * avg_r
        exp_g = 1.4 + 0.8 * avg_g
        exp_b = 1.4 + 0.8 * avg_b
        vertices_b_r = u_vertices ** (1 / exp_r)
        vertices_b_g = u_vertices ** (1 / exp_g)
        vertices_b_b = u_vertices ** (1 / exp_b)
        vertices_b = torch.cat([vertices_b_r, vertices_b_g, vertices_b_b], dim=1)

        exp_r = 2.2 - 0.8 * avg_r
        exp_g = 2.2 - 0.8 * avg_g
        exp_b = 2.2 - 0.8 * avg_b
        vertices_d_r = u_vertices ** exp_r
        vertices_d_g = u_vertices ** exp_g
        vertices_d_b = u_vertices ** exp_b
        vertices_d = torch.cat([vertices_d_r, vertices_d_g, vertices_d_b], dim=1)

        vertices_m = mid_curve(u_vertices.repeat(1, self.nc, 1))

        out = ailut_transform(img, LUTs_b, vertices_b) * self.bright_prob(img) + \
              ailut_transform(img, LUTs_m, vertices_m) * self.mid_prob(img) + \
              ailut_transform(img, LUTs_d, vertices_d) * self.dark_prob(img)

        return out


class BrightProb(nn.Module):
    def __init__(self, threshold=0.75):
        super(BrightProb, self).__init__()
        self.t = threshold

    def forward(self, x):
        return torch.clamp((x - self.t) / (1 - self.t), min=0.0, max=1.0)


class DarkProb(nn.Module):
    def __init__(self, threshold=0.25):
        super(DarkProb, self).__init__()
        self.t = threshold

    def forward(self, x):
        return torch.clamp((x - self.t) / (0 - self.t), min=0.0, max=1.0)


class MidProb(nn.Module):
    def __init__(self, threshold_b=0.75, threshold_d=0.25):
        super(MidProb, self).__init__()
        self.prob_b = BrightProb(threshold=threshold_b)
        self.prob_d = DarkProb(threshold=threshold_d)

    def forward(self, x):
        return torch.clamp(1 - self.prob_d(x) - self.prob_b(x), min=0.0, max=1.0)


class GlobPriorEncoder(nn.Module):
    def __init__(self, in_nc=3, out_nc=128):
        super(GlobPriorEncoder, self).__init__()
        self.enocoder = nn.Sequential(
            nn.Conv2d(in_nc, 16, 1),
            BasicResBlock(16),
            nn.Conv2d(16, 32, 1),
            BasicResBlock(32),
            nn.Conv2d(32, 64, 1),
            BasicResBlock(64),
            nn.Conv2d(64, out_nc, 1),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):  # shape: [b, c(3), h, w]
        return self.enocoder(x).squeeze(-1).squeeze(-1)  # shape: [b, out_nc]


class Classifier(nn.Module):
    def __init__(self, in_nc=128, out_nc=5):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_nc, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(128, out_nc),
        )

    def forward(self, x):  # shape: [b, in_nc]
        return self.classifier(x)  # shape: [b, out_nc]


class BasicResBlock(nn.Module):
    def __init__(self, nc, kernel_size=3, stride=1, padding=1, norm=False):
        super(BasicResBlock, self).__init__()
        self.conv0 = nn.Conv2d(nc, nc, kernel_size, stride=stride, padding=padding)
        self.conv1 = nn.Conv2d(nc, nc, kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(nc, affine=True) if norm else None

    def forward(self, x):
        res = self.act(self.conv0(x))
        res = self.act(self.conv1(res))
        return x + res


class LUTGenAndFuse(nn.Module):
    def __init__(self, in_fea_nc, nc, n_dim, n_lut, device) -> None:
        super().__init__()

        self.weights_gen = Classifier(in_nc=in_fea_nc, out_nc=n_lut)

        self.basic_luts = nn.Parameter(torch.zeros([n_lut, nc, n_dim, n_dim, n_dim]), requires_grad=True)
        # self.basic_luts = nn.Parameter(torch.zeros([n_lut, nc, n_dim, n_dim, n_dim]), requires_grad=False)

        self.n_lut = n_lut
        self.n_dim = n_dim
        self.nc = nc
        self.device = device

    def init_weights(self):
        for i in range(self.n_lut):
            lut = lut_from_file(str(i) + '_' + str(self.n_dim) + '.cube', dim=self.n_dim, device=self.device)
            self.basic_luts.data[i, :, :, :, :].copy_(lut)

    def forward(self, x):

        weights = torch.sigmoid(self.weights_gen(x))  # shape: [b, n_lut]
        luts = self.basic_luts  # shape: [n_lut, nc, n_dim, n_dim, n_dim]
        fused_lut = []
        for i in range(x.shape[0]):  # same fused_lut for each batch
            fused_lut_b = torch.zeros([self.nc, self.n_dim, self.n_dim, self.n_dim]).to(self.device)
            for j in range(self.n_lut):
                fused_lut_b += weights[i, j] * luts[j, :, :, :, :]  # shape: [nc, n_dim, n_dim, n_dim]
            fused_lut.append(fused_lut_b)
        fused_lut = torch.stack(fused_lut, dim=0)
        return fused_lut  # shape [b, nc, n_dim, n_dim, n_dim]

    def return_lut(self):  # for TPAMI 3D-LUT regs, see 'moudules/customize_loss.py/LutReg'
        return self.basic_luts.detach()


# this function will not be called when testing
def lut_from_file(file_name, dim, device):
    file = open(file_name, 'r')
    lines = file.readlines()
    buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

    for i in range(0, dim):
        for j in range(0, dim):
            for k in range(0, dim):
                # n = i * dim * dim + j * dim + k  # standard
                n = k * dim * dim + j * dim + i  # ieverse

                #  standard  inverse
                #   R G B     R G B
                #   0 0 0     0 0 0
                #   1 0 0     0 0 1
                #   2 0 0     0 0 2
                #   0 1 0     0 1 0
                #   1 1 0     0 1 1
                #   2 1 0     0 1 2
                #   0 2 0     0 2 0
                #     ...       ...
                #   2 2 2     2 2 2

                x = lines[n].split()
                buffer[0, i, j, k] = float(x[0])
                buffer[1, i, j, k] = float(x[1])
                buffer[2, i, j, k] = float(x[2])

    # return nn.Parameter(torch.from_numpy(buffer).requires_grad_(True)).to(device)
    return torch.from_numpy(buffer).to(device)


def mid_curve(x):
    return ((3 * torch.pi * x) - torch.cos(3 * torch.pi * x) + 1)/(3 * torch.pi + 2)
