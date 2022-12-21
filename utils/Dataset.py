import os
import sys
import nibabel as nib

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
import numpy as np
from glob import glob
import random


# hyper paramters
W = 150
L = 30
IMG_LENGTH = 48
class LiTDataset(Dataset):

    def __init__(self, ct_dir="./data/image", seg_dir="./data/label") -> None:
        self.ct_dir = glob(ct_dir + "/*")
        self.ct_dir.sort()
        self.seg_dir = glob(seg_dir + "/*")
        self.seg_dir.sort()
    
    def __getitem__(self, index) -> tuple(torch.Tensor):
        
        ct_path = self.ct_dir[index]
        seg_path = self.seg_dir[index]

        ct = read_image(ct_path)
        seg = read_image(seg_path)
        
        seg = (seg / 125).to(torch.int8)
        
        return ct, seg

    def __len__(self) -> int:
        return len(self.ct_dir)