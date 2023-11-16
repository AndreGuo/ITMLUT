# ITMLUT
Official PyTorch implemeation of "Redistributing the Precision and Content in 3D-LUT-based Inverse Tone-mapping for HDR/WCG Display" in CVMP2023.

TO BE UPDATED

<table>
<thead>
  <tr>
    <th colspan="6">AI-3D-LUT algotithms</th>
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
    <th>#BasicLUT</th>
    <th>LUT size each</th>
    <th>Extra dimension (#)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>First AI-LUT</td>
    <td rowspan="8">Image<br>enhancement<br>/retouching<br></td>
    <td>A3DLUT</td>
    <td>20-TPAMI</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9206076" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/HuiZeng/Image-Adaptive-3DLUT" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>3&times;1</td>
    <td>3&times;33<sup>3</sup></td>
    <td>-</td>
    <td>weights (of basic LUTs)</td>
    <td rowspan="4">uniform</td>
  </tr>
  <tr>
    <td>C</td>
    <td>SA-LUT-Nets</td>
    <td>ICCV'21</td>
    <td><a href="https://ieeexplore.ieee.org/document/9710177" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>-</td>
    <td>3&times;10</td>
    <td>3&times;33<sup>3</sup></td>
    <td>category (10)</td>
    <td>weights &amp; category map</td>
  </tr>
  <tr>
    <td>E<br></td>
    <td>CLUT-Net</td>
    <td rowspan="2">MM'22</td>
    <td><a href="https://dl.acm.org/doi/10.1145/3503161.3547879" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/Xian-Bei/CLUT-Net/" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>20&times;1<br></td>
    <td>3&times;5&times;20</td>
    <td>-</td>
    <td rowspan="2">weights</td>
  </tr>
  <tr>
    <td>E</td>
    <td>F2D-LUT</td>
    <td><a href="https://dl.acm.org/doi/abs/10.1145/3503161.3548325" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/shedy-pub/I2VEnhance" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>6&times;3</td>
    <td>2&times;33<sup>2</sup></td>
    <td>channel order (3)</td>
  </tr>
  <tr>
    <td>N</td>
    <td>AdaInt</td>
    <td>CVPR'22</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/9879870" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/ImCharlesY/AdaInt" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>3&times;1</td>
    <td>3&times;33<sup>3</sup></td>
    <td rowspan="3">-</td>
    <td>weights &amp; nodes</td>
    <td>learned non-uniform</td>
  </tr>
  <tr>
    <td>N</td>
    <td>SepLUT</td>
    <td>ECCV'22</td>
    <td><a href="https://link.springer.com/chapter/10.1007/978-3-031-19797-0_12" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/ImCharlesY/SepLUT" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>no</td>
    <td>3&times;9<sup>3</sup> or 3&times;17<sup>3</sup></td>
    <td>directly 1D &amp; 3D LUTs</td>
    <td>learned non-linear by 1D LUT</td>
  </tr>
  <tr>
    <td>C<br></td>
    <td>DualBLN</td>
    <td>ACCV'22</td>
    <td><a href="https://link.springer.com/chapter/10.1007/978-3-031-26313-2_11" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/120326/DualBLN" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>5&times;1</td>
    <td>3&times;36<sup>3</sup></td>
    <td>LUT fusion map</td>
    <td rowspan="4">uniform</td>
  </tr>
  <tr>
    <td>C</td>
    <td>4D-LUT</td>
    <td>23-TIP</td>
    <td><a href="https://ieeexplore.ieee.org/document/10226494" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>-</td>
    <td>3&times;1</td>
    <td>3&times;33<sup>4</sup></td>
    <td>context (33)</td>
    <td>weights &amp; context map</td>
  </tr>
  <tr>
    <td>E</td>
    <td>Photorealistic<br>Style Transfer</td>
    <td>NLUT</td>
    <td>23-arXiv</td>
    <td><a href="https://arxiv.org/abs/2303.09170" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/semchan/NLUT/" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>2048&times;1</td>
    <td>3&times;32&times;32</td>
    <td>-</td>
    <td>weights</td>
  </tr>
  <tr>
    <td>C</td>
    <td>Video Low-light<br>enhancement<br></td>
    <td>IA-LUT</td>
    <td>MM'23</td>
    <td><a href="https://dl.acm.org/doi/10.1145/3581783.3611933" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td><a href="https://github.com/Wenhao-Li-%20777/FastLLVE" target="_blank" rel="noopener noreferrer">code</a></td>
    <td>3&times;1</td>
    <td>3&times;33<sup>4</sup></td>
    <td>intensity (33)</td>
    <td>weights &amp; intensity map</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>HDR/WCG Inverse<br>Tone-mapping</td>
    <td>ITM-LUT<br></td>
    <td>CVMP'23</td>
    <td><a href="https://arxiv.org/abs/2309.17160" target="_blank" rel="noopener noreferrer">paper</a></td>
    <td>here</td>
    <td>5&times;3</td>
    <td>3&times;17<sup>3</sup></td>
    <td>luminance probability<br>(contribution) (3)<br></td>
    <td>weights</td>
    <td>explicitly defined<br>non-uniform<br></td>
  </tr>
</tbody>
</table>
