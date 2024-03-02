
import sys
sys.path.append('../')
import os


import imageio
from util.utils import process_image, colorize_mask
from pdb import set_trace as st
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
from pathlib import Path
import gc
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
from PIL import Image
from util.data_util import *
import json
import pickle


from util.data_util import face_palette as palette

"""
for visualizing gangealing results / other dsgan prediction results using the right palatte in the paper.
"""

def vis_pred():


    result_path = '/mnt/lustre/yslan/Repo/Research/ijcv-2023/gangealing/visuals/transfer/dsgan/seg_transfer/pred_seg'

    # all_pred_results = sorted(glob.glob(f'{result_path}/*.png'), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    all_pred_results = sorted(glob.glob(f'{result_path}/*.png'))

    vis = []
    for img_path in all_pred_results:
        print(img_path)
        y_pred_mask = np.array(Image.open(img_path))

        curr_vis =colorize_mask(y_pred_mask[..., 0], palette) 
        vis.append(curr_vis)

    vis = np.concatenate(vis, 1)

    # imageio.imwrite(   os.path.join('.', "gangealing_results.png"), vis)
    imageio.imwrite(   os.path.join('.', "gt_wohair.png"), vis)

def vis_gt():
    result_path = '/mnt/lustre/yslan/Repo/Research/CVPR22_REJ/datasetGAN/datasetGAN/dataset_release/annotation/testing_data/face_34_class'

    # all_results = sorted(glob.glob(f'{result_path}/mask_*.npy'), key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
    all_results = sorted(glob.glob(f'{result_path}/mask_*.npy'))

    vis = []
    for gt_path in all_results:
        print(gt_path)
        # y_pred_mask = np.array(Image.open(img_path))
        y_pred_mask = np.load(gt_path)

        ignore_hair = [
            face_class.index(name) for name in [
                'background', 
                'head***hair',
                'head***hair***sideburns', 
                'head***neck',
            ]
        ]

        for class_idx in ignore_hair:
            y_pred_mask[y_pred_mask==class_idx] = 0

        curr_vis =colorize_mask(y_pred_mask, palette) 
        vis.append(curr_vis)

    vis = np.concatenate(vis, 1)

    # imageio.imwrite(   os.path.join('.', "gangealing_results.png"), vis)
    imageio.imwrite(   os.path.join('.', "gt_wohair.png"), vis)

# vis_gt()
vis_pred()