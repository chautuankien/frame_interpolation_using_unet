from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

import os
import random

from tqdm import tqdm
from natsort import natsorted
from PIL import Image

import config
args, unparsed = config.get_args()

def make_dataset(root):
    """
    Create 2D list of all frames in N folders containing 5 frames each
    :param dir: string
                root directory containing folder
    :return: 2D list descirbed above
    """
    framesPath = []
    # Find and loop all the folder in root 'dir'
        ## natsorted: change from lexicographical order to naturally order (1,11,12,2,21 to 1,2,11,12,21)
    for index, folder in enumerate(natsorted(os.listdir(root))):
        folderPath = os.path.join(root, folder)

        framesPath.append([])
        # Find and loop all the frames inside folder
        for image in sorted(os.listdir(folderPath)):
            framesPath[index].append(os.path.join(folderPath, image))

    return framesPath


class UCF101(Dataset):
    def __init__(self, root, is_training=True):
        # framesPath = make_new_dataset(root)
        framesPath = make_dataset(root)
        if len(framesPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of : " + root + "\n"))

        self.root = root
        self.framesPath = framesPath
        self.is_training = is_training

        if self.is_training:
            
            # self.train_transforms = transforms.Compose([
            #      transforms.ToTensor(),
            #     transforms.Resize(size=(args.height, args.width)),
            #     transforms.RandomHorizontalFlip(args.horizon),
            #     transforms.RandomVerticalFlip(args.vertical),
            #     transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)])
             
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(args.height, args.width)),
            ])
        else:
            self.tranforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(args.height, args.width)),
            ])

    def __getitem__(self, index):
        # Load images
        img1 = Image.open(self.framesPath[index][0])
        img2 = Image.open(self.framesPath[index][1])
        img3 = Image.open(self.framesPath[index][2])

        # Data augmentation
        if self.is_training:
            seed = random.randint(0, 2 ** 32)
            random.seed(seed)
            img1 = self.transforms(img1)
            random.seed(seed)
            img2 = self.transforms(img2)
            random.seed(seed)
            img3 = self.transforms(img3)
            # Random Temporal Flip
            if random.random() >= 0.5:
                img1, img3 = img3, img1
        else:
            img1 = self.tranforms(img1)
            img2 = self.tranforms(img2)
            img3 = self.tranforms(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.framesPath)

def get_mean_and_std(loader):
    mean = 0.0
    std = 0.0
    total_image_count = 0
    for frames in tqdm(loader):
        #frame0, frame1, frame2, frame3, frame4 = frames
        #input = torch.cat([frame0, frame1, frame3, frame4], dim=1)
        frame0, frame1, frame2 = frames
        input = torch.cat([frame0, frame1], dim=1)
        image_count_in_a_batch = frame0.size(0)
        input = input.view(image_count_in_a_batch, frame0.size(1), -1)
        mean += input.mean(2).sum(0)
        std += input.std(2).sum(0)
        total_image_count += image_count_in_a_batch
    mean /= total_image_count
    std /= total_image_count
    print(total_image_count)
    print('mean: ' ,mean)
    print('std: ' ,std)

    return mean, std

if __name__ == "__main__":
    dir = "D:/KIEN/Dataset/UCF101/UCF101_Dataset/train1/"
    #dir = "D:/KIEN/program_python/dataset_kobayashi/ucf101_triplets/train1/"
    #transform = transforms.Compose([transforms.ToTensor()])
    trainset = UCF101(root=dir, is_training=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)

    #get_mean_and_std(trainloader)