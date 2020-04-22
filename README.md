# SIFT Reconstruction

(updating)

An official implementation code for paper "Privacy Leakage of SIFT Features via Deep Generative Model based Image Reconstruction"

## Table of Contents

- [Background](#background)

## Background
In this work, we thoroughly evaluate the privacy leakage of Scale Invariant Feature Transform (SIFT).

We propose a novel end-to-end, coarse-to-fine deep generative model for reconstructing the latent image from its SIFT features. The designed deep generative model consists of two networks, where the first one attempts to learn the structural information of the latent image by transforming from SIFT features to Local Binary Pattern (LBP) features, while the second one aims to reconstructs the pixel values guided by the learned LBP.

<p align='center'>  
  <img src='' width='870'/>
</p>

We investigate the challenging cases where the adversary can only access partial SIFT features (either descriptors or coordinates). In the case that the SIFT coordinates are not accessible, we design two methods for predicting the missing coordinate information, which achieves modest success for highly-structured images (e.g., faces). 

We demonstrate that the reconstruction performance is greatly degraded when coordinates are missing, especially for those images without regular structures. Our results would suggest that the privacy leakage problem can be largely avoided if the SIFT coordinates can be well protected.
