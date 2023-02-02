# Date: 2022-12-19
# Author: Lu Jiqiao, George
# Department : Polyu HTI
# ==============================================================================
'''This script convert data from nii file to png format slices for traning and validation'''

import os
from glob import glob
import numpy as np
import torch
import re
import cv2
import nibabel as nib
import tqdm
from torchvision.utils import save_image
from util import *
import gdown
import zipfile



def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

def seg_filter(ct:np.array, seg:np.array) -> np.array:
    ct_len = ct.shape[0]
    seg_len = seg.shape[0]
    assert ct_len == seg_len    
    mask = []
    for idx in range(ct_len):
        seg_pic = seg[idx]
        label_n = np.unique(seg_pic).shape[0]
        # at least two labels
        if label_n >= 2 and label_n <= 3:
            mask.append(idx)
    ct = ct[mask]
    seg = seg[mask]
    return ct, seg

def load_save(ct_paths, seg_paths, ct_dest, seg_dest) -> None:
    index = 1
    if not os.path.isdir(ct_dest):
        os.mkdir(ct_dest)
    if not os.path.isdir(seg_dest):
        os.mkdir(seg_dest)
    for cp, sp in zip(ct_paths, seg_paths):
        ct_scan = swap_chan(read_nii(cp))
        seg_scan = swap_chan(read_nii(sp))
        #convert label to uint8
        seg_scan = seg_scan.astype(np.uint8)
        #filter the seg so that at least images contains liver
        ct_scan, seg_scan = seg_filter(ct_scan, seg_scan)

        #rescale the image
        ct_scan = windowed(ct_scan, "ct")
        seg_scan = windowed(seg_scan, "seg")
        # name = os.path.split(cp)[1]
        # index = int(re.split("[-.]", name)[1])
        for i in range(len(ct_scan)):
            ct = ct_scan[i][...,np.newaxis]
            cv2.imwrite(os.path.join(ct_dest, "ct_image%05d.png" % index), ct)
            seg = seg_scan[i][...,np.newaxis]
            cv2.imwrite(os.path.join(seg_dest, "ct_seg%05d.png" % index), seg)
            index += 1
        # os.remove(cp)
        # os.remove(sp)
        print("reloaded: " + cp)
        
def run():
    move_vol("ct")
    ct_file = glob("./data/ct/*")
    seg_file = glob("./data/seg/*")
    ct_file.sort(key=lambda x:int(re.split(r"[.-]", x)[2]))
    seg_file.sort(key=lambda x:int(re.split(r"[.-]", x)[2]))
    load_save(ct_file, seg_file, "./data/image", "./data/label")
    

if __name__ == "__main__":
    ID = ["1-TzpAD9JjLl1getsqKrWpSzgeIzqSQ1f", "1FHg-pTTO5Q1ytAUo1KKJHjCiW6oX3lDj"]

    for i, id in enumerate(ID):
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        url = f'https://drive.google.com/uc?id={id}'

        output = f'./data/Litpart{i+1}.zip'
        gdown.download(url, output, quiet=False)

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(r'./data')

    os.rename("./data/segmentations", "./data/seg")
    print("executing script")
    run()