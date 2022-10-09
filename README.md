# frame_interpolation_using_unet
This is an implementation of video frame interpolation using UNET model.

## Results
Results on UCF101 dataset with PSNR and SSIM.
PSNR = 27.36 dB
SSIM = 0.897

## Prerequisites
This codebase was developed and tested with pytorch 1.1 and CUDA 11.3 and Python 3.8. Install:
- [Pytorch](https://pytorch.org/get-started/previous-versions/)
For GPU, run
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
