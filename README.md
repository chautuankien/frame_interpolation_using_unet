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

## Training

**Preparing training data**

In order to train the model using the provided code, the data needs to be formated in a certain manner. The `datasets/ucf101/CreateDataset.py` uses [ffmpeg](https://www.ffmpeg.org/) to extract frames from videos.

**Training**
In the `main.py`, set the parameters (dataset path, etc) and run all the cells

or to train from cmd, run:
```
python main.py --datasetPath path\to\dataset
```


