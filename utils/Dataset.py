import os
import sys
import nibabel as nib

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
import numpy as np
from glob import glob
import random
from .util import *

# hyper paramters
W = 150
L = 30
IMG_LENGTH = 48
class LiTDataset(Dataset):

    def __init__(self, ct_dir="./data/image", seg_dir="./data/label", size=[256,256], augmentation=False) -> None:
        self.ct_dir = glob(ct_dir + "/*")
        self.ct_dir.sort()
        self.seg_dir = glob(seg_dir + "/*")
        self.seg_dir.sort()
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15)
            ])
        self.preprocessing = transforms.Resize(size)
        self.augmentation = augmentation
    
    def __getitem__(self, index):
        
        ct_path = self.ct_dir[index]
        seg_path = self.seg_dir[index]
        ct = read_image(ct_path)
        seg = read_image(seg_path)
        ct = self.preprocessing(ct)
        seg = self.preprocessing(seg)
        seg = (seg / 125).to(torch.int8)
        assert torch.unique(seg).shape[0] <= 3

        if self.augmentation:
            rand_int = torch.randint(0, 100000, (1,))
            torch.manual_seed(rand_int)
            ct = self.transform(ct)
            torch.manual_seed(rand_int)
            seg = self.transform(seg)

        return ct, seg

    def __len__(self):
        return len(self.ct_dir)