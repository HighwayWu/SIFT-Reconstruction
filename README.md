# SIFT Reconstruction

An official implementation code for paper "Privacy Leakage of SIFT Features via Deep Generative Model based Image Reconstruction"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Demo](#demo)
- [Citation](#citation)


## Background
In this work, we thoroughly evaluate the privacy leakage of Scale Invariant Feature Transform (SIFT).

We propose a novel end-to-end, coarse-to-fine deep generative model (SLI) for reconstructing the latent image from its SIFT features. The designed deep generative model consists of two networks, where the first one attempts to learn the structural information of the latent image by transforming from SIFT features to Local Binary Pattern (LBP) features, while the second one aims to reconstructs the pixel values guided by the learned LBP.

<p align='center'>  
  <img src='https://github.com/HighwayWu/SIFT_Reconstruction/blob/master/imgs/framework.jpg' width='870'/>
</p>
<p align='center'>  
  <em>Framework of SLI model.</em>
</p>

We investigate the challenging cases where the adversary can only access partial SIFT features (either descriptors or coordinates). In the case that the SIFT coordinates are not accessible, we design two methods for predicting the missing coordinate information, which achieves modest success for highly-structured images (e.g., faces). 

<p align='center'>  
  <img src='https://github.com/HighwayWu/SIFT_Reconstruction/blob/master/imgs/ref-based.jpg' width='870'/>
</p>
<p align='center'>  
  <em>Reference-based Coordinates Reconstruction (SLI-R).</em>
</p>

<p align='center'>  
  <img src='https://github.com/HighwayWu/SIFT_Reconstruction/blob/master/imgs/landmark-based.jpg' width='870'/>
</p>
<p align='center'>  
  <em>Landmark-based Coordinates Reconstruction (SLI-L).</em>
</p>

We demonstrate that the reconstruction performance is greatly degraded when coordinates are missing, especially for those images without regular structures. Our results would suggest that the privacy leakage problem can be largely avoided if the SIFT coordinates can be well protected.

## Dependency
- torch 1.1.0
- opencv 3.3.0.10
- tensorflow 1.8.0

To train or test the SLI model, please download datasets from their official websites, and put them under the `./data/` directory.
The pre-trained models should be put under the `./weights/` directory.

**Note: The pretrained weights can be downloaded from:
[Google Drive](https://drive.google.com/file/d/17O_ST3WJwbLCG1jqdaM3pqbQncdg0h5i/view?usp=sharing) or 
[Baidu Yun (Code: fu5r)](https://pan.baidu.com/s/1oeHDMGVuV1hJs0yQPHMa6g)**

## Demo

To train or test the SLI model:
```bash
python main.py [--model {SLI,SLI-L,SLI-R,SLI-B}] {train,test}
```

For example to test the SLI model:
```bash
python main.py --model SLI test
```
Then the model will reconstruct the images in the `./data/celebahq_test/` and save the results in the `./res/SLI_results/` directory.

It should be noted that the "SLI-B" means the model reconstruct the images by using only the coordinates of SIFT (binary map). For more details please refer to the paper.

## Citation

If you use this code for your research, please cite our paper
```
@ARTICLE{9393327,
author={H. {Wu} and J. {Zhou}},
journal={IEEE Transactions on Information Forensics and Security},
title={Privacy Leakage of SIFT Features via Deep Generative Model Based Image Reconstruction},
year={2021},
volume={16},
number={},
pages={2973-2985},
doi={10.1109/TIFS.2021.3070427}}
```
