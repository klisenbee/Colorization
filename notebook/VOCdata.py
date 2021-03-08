# Modified torchvision VOCSegmentation Dataset Class
# Builds iterable for torch.utils.data.DataLoader() over all images in the VOC dataset.

import os
import numpy as np
import torch
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from PIL import Image
from skimage import color

#     Useage:  trainset = vocCustom('./data', download = True/False)
#         trainset[0] -> training images, trainset[1] -> target image

class vocCustom(Dataset):
    # Modified torchvision VOCSegmentation Dataset Class
    def __init__(self, img_dir, download: bool = False, HW: tuple = (256,256), resample: int = 3):
        self.img_dir = img_dir + '/VOCdevkit/VOC2012/JPEGImages'
        self.HW = HW # Resized dimensions
        self.resample = resample # Sampling for PIL.Image.resize
    
        # 2012 VOC 11 MAY 2012
        self.url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        self.md5 = '6cd6e144f989b92b3379bac3b3de84fd'
        self.filename = 'VOCtrainval_11-May-2012.tar'
        if download:
            download_url(self.url, img_dir, self.filename, self.md5)
            with tarfile.open(os.path.join(img_dir, self.filename), "r") as tar:
                tar.extractall(path = img_dir)

        file_names = [(x.split('.'))[0].strip() for x in os.listdir(img_dir + '/VOCdevkit/VOC2012/Annotations/')]

        self.images = [os.path.join(self.img_dir, x + ".jpg") for x in file_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Open image, resize, convert to lab and return training, target images
        img_orig = Image.open(self.images[index])
        
        img_rgb_rs = np.asarray(img_orig.resize((self.HW[1],self.HW[0]), resample=self.resample))
        
        img_lab_rs = color.rgb2lab(img_rgb_rs)
        img_lab_target = color.rgb2lab(img_rgb_rs)

        img_lab_target = np.moveaxis(img_lab_target[:,:,1:], -1, 0)
        img_l_rs = img_lab_rs[:,:,0]

        img_orig = torch.Tensor(img_l_rs)[None,:,:]
        img_target = torch.Tensor(img_lab_target)
        # img_orig = (1, 1, 256, 256), img_target = (1, 2, 256, 256)
        return img_orig, img_target
