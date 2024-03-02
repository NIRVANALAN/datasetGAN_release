"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from utils.seg_utils import generate_label_argmax
from ipdb import set_trace as st
import ipdb
import sys

sys.path.append('../')
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import argparse
from pathlib import Path
import gc
import os
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
from PIL import Image
from .deeplab_utils import *
import json
import pickle


class ImageLabelDataset(Dataset):

    def __init__(
            self,
            img_path_list,
            label_path_list,
            trans,
            img_size=(128, 128),
    ):
        self.label_trans = trans
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.img_size = img_size

        self.imgsA = self._read_file(0)

    def __len__(self):
        return len(self.img_path_list)

    def transform(self, img, lbl):
        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]),
                         resample=Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = transforms.ToTensor()(img)
        img = img * 2 - 1
        return img, lbl

    # def __getitem__(self, index):
    #     im_path = self.img_path_list[index]
    #     lbl_path = self.label_path_list[index]
    #     im = Image.open(im_path)

    #     suffix = Path(lbl_path).suffix

    #     try:
    #         if suffix == '.npy':
    #             lbl = np.load(lbl_path) # npy
    #         elif suffix == '.png':
    #             lbl = np.array(Image.open(lbl_path)) # png
    #     except:
    #         print(lbl_path)

    #     if len(lbl.shape) == 3:
    #         lbl = lbl[:, :, 0]

    #     lbl = self.label_trans(lbl)
    #     lbl = Image.fromarray(lbl.astype('uint8'))
    #     im, lbl = self.transform(im, lbl)

    #     return im, lbl, im_path

    def _map_seg_label_dsgan(self, seg_label):
        # follow https://github.com/NVlabs/CoordGAN/blob/b18cb817f16b86bb1e5899bd5ee59ab62de845d0/seg_utils.py
        dsgan_parts = [
            'background', 'head', 'head***cheek', 'head***chin', 'head***ear',
            'head***ear***helix', 'head***ear***lobule',
            'head***eye***botton lid', 'head***eye***eyelashes',
            'head***eye***iris', 'head***eye***pupil', 'head***eye***sclera',
            'head***eye***tear duct', 'head***eye***top lid', 'head***eyebrow',
            'head***forehead', 'head***frown', 'head***hair',
            'head***hair***sideburns', 'head***jaw', 'head***moustache',
            'head***mouth***inferior lip', 'head***mouth***oral comisure',
            'head***mouth***superior lip', 'head***mouth***teeth',
            'head***neck', 'head***nose', 'head***nose***ala of nose',
            'head***nose***bridge', 'head***nose***nose tip',
            'head***nose***nostril', 'head***philtrum', 'head***temple',
            'head***wrinkles'
        ]

        dsgan_filtered_value_key_reverse = {
            k: dsgan_parts.index(k)
            for k in dsgan_parts
        }

        coordgan_dsgan_lbls = {}
        # coordgan_lbls['head***mouth***superior lip'] = 1
        # coordgan_lbls['head***mouth***inferior lipp'] = 1
        # coordgan_lbls['head***mouth***oral comisure'] = 1
        # coordgan_lbls['head***mouth***teeth'] = 1

        # coordgan_lbls['skin'] = 6
        coordgan_dsgan_lbls['head'] = 6

        for key in dsgan_parts:
            if 'mouth' in key:
                coordgan_dsgan_lbls[key] = 1

            if 'eye*' in key:
                coordgan_dsgan_lbls[key] = 2

            if 'eyebrow' in key:
                coordgan_dsgan_lbls[key] = 3

            if 'ear' in key:
                coordgan_dsgan_lbls[key] = 4

            if 'nose' in key:
                coordgan_dsgan_lbls[key] = 5
 
            if len(key.split('***'))==2 and key.split('***')[1] not in ('nose', 'eyebrow', 'ear', 'mouth'):
                coordgan_dsgan_lbls[key] = 6

        # coordgan_dsgan_lbls['head***cheek'] = 5

        # coordgan_dsgan_lbls['head***chin'] = 6
        # coordgan_dsgan_lbls['head***jaw'] = 6
        # coordgan_dsgan_lbls['head***chin'] = 6
        # coordgan_dsgan_lbls['head***temple'] = 6
        # coordgan_dsgan_lbls['head***philtrum'] = 6
        # coordgan_dsgan_lbls['head***moustache'] = 6
        # coordgan_dsgan_lbls['head***frown'] = 6
        # coordgan_dsgan_lbls['head***forehead'] = 6
        # coordgan_dsgan_lbls['head***wrinkles'] = 6

        dsgan_filted_parts = list(coordgan_dsgan_lbls.keys())

        seg_img = np.zeros_like(seg_label)

        for idx in range(len(dsgan_filted_parts)):
            part_name = dsgan_filted_parts[idx]
            seg_part_index = dsgan_filtered_value_key_reverse[part_name]
            seg_label_part_indices = seg_label == seg_part_index
            seg_img[seg_label_part_indices] = coordgan_dsgan_lbls[part_name]

        return torch.from_numpy(seg_img)

    def _read_file(self, index):

        # * ========== load points 
        img_path = self.img_path_list[index]
        label_path = self.label_path_list[index]
        image = Image.open(img_path)

        suffix = Path(label_path).suffix

        try:
            if suffix == '.npy':
                label = np.load(label_path)  # npy
            elif suffix == '.png':
                label = np.array(Image.open(label_path))  # png
        except:
            print(label_path)

        if len(label.shape) == 3:
            label = label[:, :, 0]

        label = self.label_trans(label)
        label = Image.fromarray(label.astype('uint8'))
        image, label = self.transform(image, label)


        # label = self._map_seg_label_dsgan(label)

        # * ========= segmentation foreground points need to be transferred ====
        # ! unsqueeze() to match the dim of celebahq processing code
        label = label.unsqueeze(0)

        i, j = torch.where(label[0] > 0)
        points = torch.stack([j, i], -1)  # N, 2
        alpha_channels = torch.ones_like(points)[:, 0].float()  # N, 1

        label_colors = (label[:, i, j]).squeeze().reshape(1, 1, -1,
                                                          1).float()  # 1 C H W

        # * seg label apping

        label_colors_colorize = generate_label_argmax(
            label_colors, n_label=34).squeeze().permute(1, 0).float()  # N 3
        
        seg_label_colors = label_colors.reshape(-1)

        return {
            "imgsB": image,
            "labelB": (label * 255).float(),
            "imgsB_path": img_path,
            "labelsB_path": label_path,
            "pointsB": points,
            'label_colors': label_colors_colorize,
            'label_pointsB': seg_label_colors,
            'alpha_channels': alpha_channels,
        }

        # return im, lbl, im_pa

    def __getitem__(self, index):

        imgsB = self._read_file(index)
        return {
            **imgsB,
            "imgsA": self.imgsA['imgsB'],
            "pointsA": self.imgsA['pointsB'],
            "labelA": self.imgsA['labelB'],
            "imgsA_path": self.imgsA['imgsB_path'],
            "labelsA_path": self.imgsA['labelsB_path'],
            'label_colors_A': self.imgsA['label_colors'],
            'alpha_channels_A': self.imgsA['alpha_channels'],
            'label_pointsA': self.imgsA['label_pointsB'],
        }


class ImageDataset(Dataset):

    def __init__(
            self,
            img_path_list,
            trans=None,
            img_size=(128, 128),
    ):
        self.label_trans = trans
        self.img_path_list = img_path_list
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        im = Image.open(im_path)

        im = self.transform(im)

        return im, im_path

    def transform(self, img):
        img = img.resize((self.img_size[0], self.img_size[1]))
        img = transforms.ToTensor()(img)
        return img