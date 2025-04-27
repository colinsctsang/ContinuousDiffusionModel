# Continuous Diffusion Model for Self-supervised Denoising and Super-resolution on Fluorescence Microscopy Images
This is the official Pytorch implementation of ***"Continuous Diffusion Model for Self-supervised Denoising and Super-resolution on Fluorescence Microscopy Images" (IPMI 2025)***, written by Colin S. C. Tsang and Albert C. S. Chung.

## Prerequisites
This code was tested with `Pytorch 1.13.1` and NVIDIA GeForce RTX 4080 with 16GB memory.

## Training and testing scripts
- `train.py`: Train the Continuous Diffusion Model in a <u>self-supervised</u> manner.

- `test.py`: Test the model and evaluate it in RMSE and SSIM. 

## Dataset
We use the dataset from https://github.com/IVRL/w2s.

## Publication
If you find this repository useful, please cite:
- **Continuous Diffusion Model for Self-supervised Denoising and Super-resolution on Fluorescence Microscopy Images**  
Colin S. C. Tsang and Albert C. S. Chung  
IPMI 2025


## Acknowledgment
Some codes in this repository are modified/copied from https://github.com/BUPTLdy/Pytorch-LapSRN and https://github.com/yinboc/liif

The SSIM function is provided by https://github.com/jacenfox/pytorch-msssim

###### Keywords
Keywords: Super-resolution, Denoising, Self-supervised.
