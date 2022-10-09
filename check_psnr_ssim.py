import cv2
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

def check_psnr_ssim_one_image(path, number):

    gt = cv2.imread(path + str(number) + "_gt.png")
    out = cv2.imread(path + str(number) + "_out.png")

    PSNR = peak_signal_noise_ratio(gt, out)
    SSIM = structural_similarity(gt, out, multichannel=True)

    print("PSNR for image" + str(number) + "is" + str(PSNR))
    print("SSIM for image" + str(number) + "is" + str(SSIM))

def check_psnr_ssim_overall(path):
    PSNR = 0.0
    SSIM = 0.0

    dataset_size = len([name for name in os.listdir(path)]) / 2
    int_dataset_size = int(dataset_size)

    for i in tqdm(range(int_dataset_size)):
        if i == 4298:   # error -> skip
            continue
        gt = cv2.imread(data_path + str(i) + "_gt.png")
        out = cv2.imread(data_path + str(i) + "_out.png")

        SSIM += structural_similarity(gt, out, multichannel=True)
        PSNR += peak_signal_noise_ratio(gt, out)

    print("PSNR overall: " + str(PSNR / (dataset_size)))
    print("SSIM overall: " + str(SSIM / (dataset_size)))

if __name__ == '__main__':
    data_path = ''

    # check_psnr_ssim_one_image(path=data_path, number=2)
    check_psnr_ssim_overall(path=data_path)


