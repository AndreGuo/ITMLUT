# ITMLUT
Official PyTorch implemeation of "Redistributing the Precision and Content in 3D-LUT-based Inverse Tone-mapping for HDR/WCG Display"([paper(arViv)](https://arxiv.org/abs/2309.17160), [paper](https://dl.acm.org/doi/abs/10.1145/3626495.3626503)) in CVMP2023 ([website](https://www.cvmp-conference.org/2023/), [proceedings](https://dl.acm.org/doi/proceedings/10.1145/3626495)).

# 1. A quick glance to all AI-3D-LUT algorithms

Here are all AI-3D-LUT (look-up table) as far as we know (last updated 07/03/2024), please jump to them if interested.

You can cite our paper if you feel this helpful.

    @InProceedings{Guo_2023_CVMP,
        author    = {Guo, Cheng and Fan, Leidong and Zhang, Qian and Liu, Hanyuan and Liu, Kanglin and Jiang, Xiuhua},
        title     = {Redistributing the Precision and Content in 3D-LUT-based Inverse Tone-mapping for HDR/WCG Display},
        booktitle = {Proceedings of the 20th ACM SIGGRAPH European Conference on Visual Media Production (CVMP)},
        month     = {November},
        year      = {2023},
        pages     = {1-10},
        doi       = {10.1145/3626495.3626503}
    }

<table>
<thead>
  <tr>
    <th colspan="7">AI-3D-LUT algotithms</th>
    <th colspan="3">Expressiveness of the trained LUT</th>
    <th rowspan="2">Output of<br>neural network(s)<br></th>
    <th rowspan="2">Nodes<br>(packing)<br></th>
  </tr>
  <tr>
    <th>Idea</th>
    <th>Task</th>
    <th>Name<br></th>
    <th>Publication</th>
    <th>Paper<br></th>
    <th>Code</th>
    <th>Institution</th>
    <th>#BasicLUT</th>
    <th>LUT size each</th>
    <th>(#) Extra dimension</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>First AI-LUT</td>
    <td rowspan="9">Image<br>enhancement<br>/retouching<br></td>
    <td><b>A3DLUT</b></td>
    <td>20-TPAMI</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9206076" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/HuiZeng/Image-Adaptive-3DLUT" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>HK_PolyU &amp; <a href="https://www.dji.com/" target="_blank" rel="noopener noreferrer">DJI Innovation</a></td>
    <td>3&times;1</td>
    <td>3&times;33<sup>3</sup></td>
    <td>-</td>
    <td>weights (of basic LUTs)</td>
    <td rowspan="4">uniform</td>
  </tr>
  <tr>
    <td>C</td>
    <td><b>SA-LUT-Nets</b></td>
    <td>ICCV'21</td>
    <td><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Real-Time_Image_Enhancer_via_Learnable_Spatial-Aware_3D_Lookup_Tables_ICCV_2021_paper.pdf" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>-</td>
    <td><a href="https://www.noahlab.com.hk/" target="_blank" rel="noopener noreferrer">Huawei Noah's Ark Lab</a></td>
    <td>3&times;<b>10</b></td>
    <td>3&times;33<sup>3</sup></td>
    <td>(<b>10</b>) category</td>
    <td>weights &amp; category map</td>
  </tr>
  <tr>
    <td>E<br></td>
    <td><b>CLUT-Net</b></td>
    <td rowspan="2">MM'22</td>
    <td><a href="https://dl.acm.org/doi/10.1145/3503161.3547879" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/Xian-Bei/CLUT-Net/" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>CN_TongjiU &amp; <a href="https://ur.oppo.com/" target="_blank" rel="noopener noreferrer">OPPO Research</a></td>
    <td>20&times;1<br></td>
    <td>3&times;5&times;20 (compressed LUT representation)</td>
    <td>-</td>
    <td rowspan="2">weights</td>
  </tr>
  <tr>
    <td>E</td>
    <td><b>F2D-LUT</b></td>
    <td><a href="https://dl.acm.org/doi/abs/10.1145/3503161.3548325" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/shedy-pub/I2VEnhance" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>CN_TsinghuaU</td>
    <td>6&times;<b>3</b></td>
    <td>2&times;33<sup>2</sup> (3D LUT decoupled to 2D LUTs) </td>
    <td>(<b>3</b>) R-G/R-B/G-B channel order</td>
  </tr>
  <tr>
    <td>N</td>
    <td><b>AdaInt</b></td>
    <td>CVPR'22</td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_AdaInt_Learning_Adaptive_Intervals_for_3D_Lookup_Tables_on_Real-Time_CVPR_2022_paper.pdf" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/ImCharlesY/AdaInt" target="_blank" rel="noopener noreferrer">code</a></td>
    <td rowspan="2">CN_SJTU &amp; Alibaba Group</td>
    <td>3&times;1</td>
    <td>3&times;33<sup>3</sup></td>
    <td rowspan="3">-</td>
    <td>weights &amp; nodes</td>
    <td>learned non-uniform</td>
  </tr>
  <tr>
    <td>N</td>
    <td><b>SepLUT</b></td>
    <td>ECCV'22</td>
    <td><a href="https://link.springer.com/content/pdf/10.1007/978-3-031-19797-0_12" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/ImCharlesY/SepLUT" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>1 (no self-adaptibility)</td>
    <td>3&times;9<sup>3</sup> or 3&times;17<sup>3</sup></td>
    <td>directly 1D &amp; 3D LUTs</td>
    <td>learned non-linear by 1D LUT</td>
  </tr>
  <tr>
    <td>C<br></td>
    <td><b>DualBLN</b></td>
    <td>ACCV'22</td>
    <td><a href="https://openaccess.thecvf.com/content/ACCV2022/papers/Zhang_DualBLN_Dual_Branch_LUT-aware_Network_for_Real-time_Image_Retouching_ACCV_2022_paper.pdf" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/120326/DualBLN" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>CN_NorthwesternPolyU</td>
    <td>5&times;1</td>
    <td>3&times;36<sup>3</sup></td>
    <td>LUT fusion map</td>
    <td rowspan="7">uniform</td>
  </tr>
  <tr>
    <td>C</td>
    <td><b>4D-LUT</b></td>
    <td>23-TIP</td>
    <td><a href="https://ieeexplore.ieee.org/document/10226494" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>-</td>
    <td>CN_XianJiaotongU &amp; <a href="https://www.msra.cn" target="_blank" rel="noopener noreferrer">Microsoft Research Asia</a></td>
    <td>3&times;1</td>
    <td>3&times;33<sup><b>4</b></sup></td>
    <td>(<b>33</b>) context</td>
    <td>weights &amp; context map</td>
  </tr>
  <tr>
    <td>C &amp E</td>
    <td><b>AttentionLUT</b></td>
    <td>24-ArXiv</td>
    <td><a href="https://arxiv.org/pdf/2401.01569.pdf" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>-</td>
    <td>CN_XianJiaotongU</td>
    <td>no (donot relay on basic LUT for self-adaptibility)</td>
    <td>9&times;15&times;33 (represented by Canonical Polyadic tensor decomposition)</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>E</td>
    <td>Photorealistic<br>Style Transfer</td>
    <td><b>NLUT</b></td>
    <td>23-arXiv</td>
    <td><a href="https://arxiv.org/pdf/2303.09170" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/semchan/NLUT/" target="_blank" rel="noopener noreferrer">code</a></td>
    <td><a href="http://international.sobey.com/index.php" target="_blank" rel="noopener noreferrer">Sobey Digital Technology</a> &amp; Peng Cheng Lab</td>
    <td>2048&times;1</td>
    <td>3&times;32&times;32 (compressed LUT representation)</td>
    <td>-</td>
    <td>weights</td>
  </tr>
  <tr>
    <td>C</td>
    <td>Video Low-light<br>enhancement<br></td>
    <td><b>IA-LUT</b></td>
    <td>MM'23</td>
    <td><a href="https://dl.acm.org/doi/10.1145/3581783.3611933" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/Wenhao-Li-%20777/FastLLVE" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>CN_SJTU &amp; <a href="https://damo.alibaba.com/" target="_blank" rel="noopener noreferrer">Alibaba Damo Academy</a></td>
    <td>3&times;1</td>
    <td>3&times;33<sup><b>4</b></sup></td>
    <td>(<b>33</b>) intensity</td>
    <td>weights &amp; intensity map</td>
  </tr>
  <tr>
    <td>No</td>
    <td>Underwater Imge Enhancement</td>
    <td><b>INAM-LUT</b></td>
    <td>23-Sensors</td>
    <td><a href="https://www.mdpi.com/1424-8220/23/4/2169" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>-</td>
    <td>CN_XidianU </td>
    <td>3&times;1</td>
    <td>3&times;33(?)<sup>3</sup></td>
    <td>-</td>
    <td>weights</td>
  </tr>
  <tr>
    <td>C</td>
    <td>Tone-mapping</td>
    <td><b>LapLUT</b></td>
    <td>NeurIPS'23</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/b3a08d179347e33414badadf100e4e8d-Paper-Conference.pdf" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>-</td>
    <td>CN_HUST  &amp; <a href="https://www.dji.com/" target="_blank" rel="noopener noreferrer">DJI Innovation</a></td>
    <td>3&times;1</td>
    <td>3&times;33(?)<sup>3</sup></td>
    <td>-</td>
    <td>weight map (of each interpolated image)</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>HDR/WCG Inverse<br>Tone-mapping</td>
    <td><b>ITM-LUT</b><br></td>
    <td>CVMP'23</td>
    <td><a href="https://dl.acm.org/doi/abs/10.1145/3626495.3626503" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>see below</td>
    <td><a href="https://en.cuc.edu.cn/" target="_blank" rel="noopener noreferrer">CN_CUC</a> &amp; Peng Cheng Lab</td>
    <td>5&times;<b>3</b></td>
    <td>3&times;17<sup>3</sup></td>
    <td>(<b>3</b>) luminance probability<br>(contribution)<br></td>
    <td>weights</td>
    <td>explicitly defined<br>non-uniform<br></td>
  </tr>
</tbody>
</table>

In col. *idea*:

**C** stands for improving the expressiveness of LUT **c**ontent (by new way to generate image-adaptive LUT or introducing new dimension);

**E** stands for making LUT further **e**fficient (by special representation of LUT's elements);

**N** stands for setting non-uniform **n**odes (to optimize LUT's interpolation error on image with specific numerical distribution).

**Note that**:

We only listed AI-3D-LUTs for image-to-image low-level vision tasks, and below AI-LUTs are not included:
+ Non-3D AI-LUTs for other CV tasks: *e.g.* [SR-LUT](https://openaccess.thecvf.com/content/CVPR2021/papers/Jo_Practical_Single-Image_Super-Resolution_Using_Look-Up_Table_CVPR_2021_paper.pdf), MuLUT([paper1](https://link.springer.com/content/pdf/10.1007/978-3-031-19797-0_14), [paper2(extented to image restoration)](https://arxiv.org/pdf/2303.14506.pdf)), [VA-LUT](https://arxiv.org/pdf/2303.00334), [SPLUT](https://arxiv.org/pdf/2207.12987.pdf) (**super-resolution**, non-3D-LUT), [MEFLUT](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_MEFLUT_Unsupervised_1D_Lookup_Tables_for_Multi-exposure_Image_Fusion_ICCV_2023_paper.pdf) (**multi-exposure fusion**, 1D-LUT), [SA-LuT-Nets](https://link.springer.com/content/pdf/10.1007/978-3-030-59719-1_22.pdf) (**medical imaging**) *etc.* (Such LUTs may not even involve an interpolation process).
+ Claim to be AI-LUT, but use other mechanism to conduct image-to-image transform: *e.g.* [NILUT](https://arxiv.org/pdf/2306.11920.pdf) (represent LUT transform using MLP(multi-layer perceptron)) *etc*.

# 2. Our algorithm ITM-LUT

Our AI-3D-LUT alogorithm named ***ITM-LUT*** conduct inverse tone-mapping (ITM) from standard dynamic range (SDR) image/frame to its high dynamic range and wide color gamut (HDR/WCG) version.

## 2.1 Key features

- ***Self-adaptability:*** LUT content will alter with input SDR's statistics, by merging basic LUTs using neural-network-generated weight from input SDR.
- ***AI-learning:*** Rather a 'top-down design' static LUT, our LUT can be learned from any dataset in 'bottom-up' manner, enabling the reverse engineering of any technical and artistic intent between SDR and HDR/WCG.
- ***HDR/WCG optimization:*** For a LUT processing higher-bit-depth HDR/WCG content (requiring larger LUT size *N*), we use 3 LUTs with different non-uniform nodes. Their result will have less interpolation error respectively in different ranges, so we use a pixel-wise contribution map to blend their best ranges. In this case, 3 smaller LUTs (e.g. *N*=17) can reach the same error level to single bigger LUT (e.g. *N*=33), while occupy less #elements (e.g. 44217<107811).

## 2.2 Prerequisites

- Python
- PyTorch
- OpenCV
- ImageIO
- NumPy
- GCC/G++

## 2.3 Usage (how to test)

First, install the CUDA&C++ implementation of ***trilinear interpolation with non-uniform vertices*** (need GCC/G++):

```bash
python3 ./ailut/setup.py install
```
after that, you can get `ailut` package in your python.

Run `test.py` with below configuration(s):

```bash
python3 test.py frameName.jpg
```

When batch processing, use wildcard `*`:

```bash
python3 test.py framesPath/*.png
```

or like:

```bash
python3 test.py framesPath/footageName_*.png
```

Add below configuration(s) for specific propose:

| Propose                                                                                          |                                    Configuration                                     |
|:-------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------:|
| Specifing output path                                                                            |                       `-out resultDir/` (default is inputDir)                        |
| Resizing image before inference                                                                  |                       `-resize True -height newH -width newW`                        |
| Adding filename tag                                                                              |                                    `-tag yourTag`                                    |
| Forcing CPU processing                                                                           |                                   `-use_gpu False`                                   |
| Using input SDR with bit depth != 8                                                              |                               *e.g.* `-in_bitdepth 16`                               |
| Saving result HDR in other format<br/>(defalut is uncompressed<br/>16-bit `.tif`of single frame) | `-out_format suffix`<br>`png` as 16bit .png<br>`exr` require extra package `openEXR` |

Change `line 104` in `test.py` to use other parameters/checkpoint:

+ Current `params.pth` is trained on our own ***HDRTV4K*** dataset and ***DaVinci*** degradation model (available [here](https://github.com/AndreGuo/HDRTVDM/#DaVinciSDR)).
+ Checkpoint `params_TV1K.pth` is trained on popular ***HDRTV1K*** dataset and ***YouTube*** degradation model, it can score 36.14dB the PSNR, 0.9784 the SSIM, 11.447 the $\Delta$ E<sub>itp</sub> and 8.8070 VDP3 ('task'='side-by-side', 'color_encoding'='rgb-bt.2020', 'pixel_per_degree'=60 on 1920*1080 image) on ***HDRTV1K*** testset.
+ We will later release more interesting checkpoint(s).

## 2.4 Training code

First, download the training code from [BaiduNetDisk](https://pan.baidu.com/s/18Zw6UJsrAZHw1Hy6cYiPSQ?pwd=2nzj)(code:2nzj) or [GoogleDrive](https://drive.google.com/file/d/1zhIJ3igFIPgmWK9BiBAh5TSYqL4GL7wL/view?usp=drive_link).
This package contain 5 essential real ITM LUTs used in our own LUT initialization, and other 13 real ITM LUTs (in both N=17/33/65) where you can use any of their combinations to try new LUT initialization.

Then:

```bash
cd ITMLUT_train/codes
```

```bash
python3 train.py -opt options/test/test_Net.yml
```

You can modify training configuration e.g. #basicLUTs and LUTsize at `codes/options/test/test_Net.yml`.

## 2.5 Changelog

| Date          |   log  |
|:-------------:|:------:|
| 29 Feb 2024   | Since most SoTAs are still trained and tested on ***HDRTV1K*** dataset, we add a checkpoint `params_TV1K.pth` trained on it, so result will get a similar look as SoTAs. |
| 3 Mar 2024 | Training code (along with 18 real ITM LUTs in N=17/33/65) is now released. |

## Contact

Guo Cheng ([Andre Guo](https://orcid.org/orcid=0000-0002-2660-2267)) guocheng@cuc.edu.cn

- *State Key Laboratory of Media Convergence and Communication (MCC),
Communication University of China (CUC), Beijing, China.*
- *Peng Cheng Laboratory (PCL), Shenzhen, China.*
