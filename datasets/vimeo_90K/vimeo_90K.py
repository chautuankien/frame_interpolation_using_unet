import os.path
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

"""
def make_dataset(root, list_file):
    raw_im_list = open(os.path.join(root, list_file)).read().splitlines()
    # the last line is invalid in test set.
    # print("The last sample is : " + raw_im_list[-1])
    raw_im_list = raw_im_list[:-1]
    assert len(raw_im_list) > 0
    random.shuffle(raw_im_list)

    return raw_im_list

def Vimeo_90K(root, split=1.0, single=False, task = 'interp' ):
    train_list = make_dataset(root, "tri_trainlist.txt")
    test_list = make_dataset(root, "tri_testlist.txt")
    train_dataset = ListDataset(root, train_list, loader=Vimeo_90K_loader)
    test_dataset = ListDataset(root, test_list, loader=Vimeo_90K_loader)
    return train_dataset, test_dataset
"""

class Vimeo_90K(Dataset):
    def __init__(self, root, is_training):
        self.root = root
        self.image_root = os.path.join(self.root, 'sequences')
        self.is_training = is_training

        train_fn = os.path.join(self.root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if self.is_training:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.is_training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        # img1 = Image.open(imgpaths[0])
        # img2 = Image.open(imgpaths[1])
        # img3 = Image.open(imgpaths[2])
        images = [Image.open(img) for img in imgpaths]

        # Data augmentation
        if self.is_training:
            seed = random.randint(0, 2 ** 32)
            images_ = []
            for img_ in images:
                random.seed(seed)
                images.append(self.transforms(img_))
            images = images_
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                # img1, img3 = img3, img1
                # imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
        else:
            # T = transforms.ToTensor()
            # img1 = T(img1)
            # img2 = T(img2)
            # img3 = T(img3)
            images = [self.transforms(img_) for img_ in images]

        return images

    def __len__(self):
        if self.is_training:
            return len(self.trainlist)
        else:
            return len(self.testlist)

class VimeoSepTuplet(Dataset):
    def __init__(self, root, is_training, input_frames="1357", mode='mini'):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        Outputs
            frames: list of 4 frames
            gt : grouth truth frames
        """
        self.root = root
        self.image_root = os.path.join(self.root, 'sequences')
        self.is_training = is_training
        self.inputs = input_frames

        train_fn = os.path.join(self.root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        # reduce the number of test images if mode == 'mini'
        if mode != 'full':
            tmp = []
            for i, value in enumerate(self.testlist):
                if i % 38 == 0:
                    tmp.append(value)
            self.testlist = tmp

        if self.is_training:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        if self.is_training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        imgpaths = [imgpath + f'/im{i}.png' for i in range(1, 8)]

        # Load images
        images = [Image.open(img) for img in imgpaths]

        # Select relevant inputs, 0th, 2th, 4th, 6th
        inputs = [int(e)-1 for e in list(self.inputs)]  # inputs = [0,2,4,6]
        inputs = inputs[:len(inputs)//2] + [3] + inputs[len(inputs)//2:]  # inputs = [0,2,3,4,6]
        images = [images[i] for i in inputs]
        imgpaths = [imgpaths[i] for i in inputs]

        # Data augmentation
        if self.is_training:
            seed = random.randint(0, 2**32)
            images_ = []
            for img_ in images:
                random.seed(seed)
                images_.append(self.transforms(img_))
            images = images_

            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]
            gt = images[len(images)//2]
            images = images[:len(images)//2] + images[len(images)//2+1:]

            return images, gt
            # return images
        else:
            images = [self.transforms(img_) for img_ in images]
            gt = images[len(images)//2]
            images = images[:len(images)//2] + images[len(images)//2+1:]

            return images, gt
            # return images

    def __len__(self):
        if self.is_training:
            return len(self.trainlist)
        else:
            return len(self.testlist)

def get_loader(mode, data_root, batch_size, shuffle, num_workers):
    if mode == 'train':
        is_training =True
    else:
        is_training = False
    dataset = VimeoSepTuplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)


