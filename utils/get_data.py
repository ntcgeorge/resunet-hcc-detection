# Date: 2022-12-19
# Author: Lu Jiqiao, George
# Department : Polyu HTI
# ==============================================================================
'''This script create data as png format for traning and validation'''

import os
from glob import glob
import numpy as np
import torch
import re
import cv2
import nibabel as nib
import tqdm

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

def move_vol(dest_dir:str):
  vol_list = glob(r"./data/volume_pt*/*.nii")
  os.makedirs(f"./data/{dest_dir}")
  for file in vol_list:
    temp = file.split(r"/")
    temp[2] = dest_dir
    dest = "/".join(temp)
    os.rename(file, dest)

def swap_resize(array:np.array):#shape(c,512,512)
  n_dim = len(array.shape)
  if n_dim != 3:
    raise ValueError("the dimention should be 3")
  dim = array.shape[2]
  container = np.zeros(dim * 128* 128).reshape((dim, 128, 128))
  for i in range(dim):
    container[i] = cv2.resize(array[...,i], (128, 128), interpolation=cv2.INTER_AREA)
  return container

def windowed(arr:np.array, mode:str, w=150, l=30):
    if mode == 'ct':
        px = arr
        px_min = l - w//2
        px_max = l + w//2
        px[px<px_min] = px_min
        px[px>px_max] = px_max
        return (px - px_min) * 255. / (px_max - px_min) 
    elif mode == "seg":
        return arr * 255. / 2.
    else:
        raise ValueError(f"the mode should be seg or ct but received {mode}")
    

def load_save(ct_paths, seg_paths, ct_dest, seg_dest) -> None:
    index = 1
    if not os.path.isdir(ct_dest):
        os.mkdir(ct_dest)
    if not os.path.isdir(seg_dest):
        os.mkdir(seg_dest)
    for cp, sp in zip(ct_paths, seg_paths):
        ct_scan = swap_resize(read_nii(cp))
        seg_scan = swap_resize(read_nii(sp))
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
            cv2.imwrite(os.path.join(ct_dest, "ct_image%05d.jpg" % index), ct)
            seg = seg_scan[i][...,np.newaxis]
            cv2.imwrite(os.path.join(seg_dest, "ct_seg%05d.jpg" % index), seg)
            index += 1
        # os.remove(cp)
        # os.remove(sp)
        print("reloaded: " + cp)
        
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
def main():
    move_vol("ct")
    ct_file = glob("./data/ct/*")
    seg_file = glob("./data/seg/*")
    ct_file.sort(key=lambda x:int(re.split(r"[.-]", x)[2]))
    seg_file.sort(key=lambda x:int(re.split(r"[.-]", x)[2]))
    load_save(ct_file, seg_file, "./data/image", "./data/label")
    

if __name__ == "__main__":
    print("executing script")
    main()