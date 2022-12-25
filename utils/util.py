# Date: 2022-12-19
# Author: Lu Jiqiao, George
# Department : Polyu HTI
# ==============================================================================
'''This utils provides a series of function for pre-processing and displaying'''
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def display(img, label, index, color_map='nipy_spectral'):
    '''
    display the mask, label and ovelapping of them
    '''
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='bone')
    plt.axis('off')
    plt.title(f'Windowed Image case{index}', alpha=0.5)

    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap='bone')
    plt.axis('off')
    plt.title(f'Mask case{index}')

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='bone')
    plt.imshow(label,cmap=color_map, alpha=0.5)
    plt.axis('off')
    plt.title(f'Liver and Mask case{index}')

def move_vol(dest_dir:str):
    vol_list = glob(r"./data/volume_pt*/*.nii")
    os.makedirs(f"./data/{dest_dir}")
    for file in vol_list:
        temp = file.split(r"/")
        temp[2] = dest_dir
        dest = "/".join(temp)
        os.rename(file, dest)

def swap_chan(array:np.array): #shape(c, 512, 512)
    n_dim = array.ndim
    if n_dim != 3:
        raise ValueError("the input dimension should be 3")
    return np.transpose(array, (2,0,1))

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

def get_local_data(path:str):
    file = Image.open(path)
    image = np.asarray(file)
    image = windowed(image, mode="ct")


    