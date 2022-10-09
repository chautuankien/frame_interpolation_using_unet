import torch.utils.data as data
import os
import os.path
from cv2 import imread
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def Vimeo_90K_loader(root, im_path, input_frame_size = (3, 256, 448), output_frame_size = (3, 256, 448), data_aug = True):
    root = os.path.join(root, 'sequences', im_path)

    if data_aug and random.randint(0, 1):
        path_pre2 = os.path.join(root,  "im1.png")
        path_mid = os.path.join(root,  "im2.png")
        path_pre1 = os.path.join(root,  "im3.png")
    else:
        path_pre1 = os.path.join(root,  "im1.png")
        path_mid = os.path.join(root,  "im2.png")
        path_pre2 = os.path.join(root,  "im3.png")

    im_pre2 = imread(path_pre2)
    im_pre1 = imread(path_pre1)
    im_mid = imread(path_mid)

    h_offset = random.choice(range(256 - input_frame_size[1] + 1))
    w_offset = random.choice(range(448 - input_frame_size[2] + 1))

    im_pre2 = im_pre2[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
    im_pre1 = im_pre1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
    im_mid = im_mid[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]

    if data_aug:
        if random.randint(0, 1):
            im_pre2 = np.fliplr(im_pre2)
            im_mid = np.fliplr(im_mid)
            im_pre1 = np.fliplr(im_pre1)
        if random.randint(0, 1):
            im_pre2 = np.flipud(im_pre2)
            im_mid = np.flipud(im_mid)
            im_pre1 = np.flipud(im_pre1)

    X0 = np.transpose(im_pre1, (2,0,1))
    X2 = np.transpose(im_pre2, (2, 0, 1))

    y = np.transpose(im_mid, (2, 0, 1))
    return X0.astype("float32")/255.0, \
            y.astype("float32")/255.0,\
            X2.astype("float32")/255.0
class ListDataset(data.Dataset):
    def __init__(self, root, path_list, loader=Vimeo_90K_loader):
        self.root = root
        self.path_list = path_list
        self.loader = loader

    def __getitem__(self, index):
        path = self.path_list[index]
        # print(path)
        image_0, image_1, image_2 = self.loader(self.root, path)
        return image_0, image_1, image_2

    def __len__(self):
        return len(self.path_list)

def Vimeo_90K_loader(root, im_path, input_frame_size = (3, 256, 448), output_frame_size = (3, 256, 448), data_aug = True):
    root = os.path.join(root, 'sequences', im_path)

    if data_aug and random.randint(0, 1):
        path_pre2 = os.path.join(root,  "im1.png")
        path_mid = os.path.join(root,  "im2.png")
        path_pre1 = os.path.join(root,  "im3.png")
    else:
        path_pre1 = os.path.join(root,  "im1.png")
        path_mid = os.path.join(root,  "im2.png")
        path_pre2 = os.path.join(root,  "im3.png")

    im_pre2 = Image.open(path_pre2)
    im_pre1 = Image.open(path_pre1)
    im_mid = Image.open(path_mid)

    h_offset = random.choice(range(256 - input_frame_size[1] + 1))
    w_offset = random.choice(range(448 - input_frame_size[2] + 1))

    im_pre2 = im_pre2[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
    im_pre1 = im_pre1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]
    im_mid = im_mid[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2], :]

    if data_aug:
        if random.randint(0, 1):
            im_pre2 = np.fliplr(im_pre2)
            im_mid = np.fliplr(im_mid)
            im_pre1 = np.fliplr(im_pre1)
        if random.randint(0, 1):
            im_pre2 = np.flipud(im_pre2)
            im_mid = np.flipud(im_mid)
            im_pre1 = np.flipud(im_pre1)

    X0 = np.transpose(im_pre1, (2,0,1))
    X2 = np.transpose(im_pre2, (2, 0, 1))

    y = np.transpose(im_mid, (2, 0, 1))
    return X0.astype("float32")/255.0, \
            y.astype("float32")/255.0,\
            X2.astype("float32")/255.0

class VimeoTriplet(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training

        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])

        # Data augmentation
        if self.training:
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
                imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
        else:
            T = transforms.ToTensor()
            img1 = T(img1)
            img2 = T(img2)
            img3 = T(img3)

        # imgs = [img1, img2, img3]

        return img1, img2, img3

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
        return 0

def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoTriplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


