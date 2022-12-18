import os
import sys
import nibabel as nib

import torch
from torch.utils.data import Dataset
import numpy as np

import random


# hyper paramters
W = 150
L = 30

class LiTDataset(Dataset):

    def __init__(self, ct_dir="./data/ct", seg_dir="./data/seg") -> None:
        self.ct_dir = os.listdir(ct_dir)
        self.seg_dir = os.listdir(seg_dir)
    
    def __getitem__(self, index):
        
        ct_path = self.ct_dir[index]
        seg_path = self.seg_dir[index]

        ct = nib.load(ct_path)
        seg = nib.load(ct_path)
        return self.windowed(ct), self.windowed(seg)

    def windowed(arr:np.array, w=W, l=L):
        px = arr
        px_min = l - w//2
        px_max = l + w//2
        px[px<px_min] = px_min
        px[px>px_max] = px_max
        return (px-px_min) / (px_max-px_min)
    
    def __len__(self):
        return len(self.ct_dir)