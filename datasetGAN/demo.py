# This file do segmentation inference over a given folder.
import sys

import cv2
from tqdm import tqdm

sys.path.append('../')
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import argparse
import gc
import glob
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import imageio
import ipdb
import matplotlib.pyplot as plt
import scipy.misc
import torch
import torchvision
from dotmap import DotMap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from util.data_util import *
from util.utils import colorize_mask, process_image

from train_deeplab import ImageDataset, ImageLabelDataset


def load_cfg(args):
    if args['category'] == 'car':
        from util.data_util import car_20_palette as palette
        if args['testing_data_number_class'] == 12:
            from util.data_util import car_12_class as class_name
        elif args['testing_data_number_class'] == 20:
            from util.data_util import car_20_class as class_name
    elif args['category'] == 'face':
        from util.data_util import face_class as class_name
        from util.data_util import face_palette as palette

    elif args['category'] == 'bedroom':
        from util.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from util.data_util import cat_palette as palette
        class_name = None

    print(args['category'])

    return class_name, palette


import ipdb


def inference(args,
              save=True,
              blur=True,
              ksize=31,
              std=33,
              class_name_to_ignore=None):

    class_name, palette = load_cfg(args)
    # ids = range(args['testing_data_number_class'])

    # data_all = glob.glob(args['testing_path'] + "**/[0-9].png",
    #                      recursive=True)  # ?
    data_all = Path(args['dataset_path']).glob('*.png')
    data_all = [str(path) for path in sorted(data_all)]
    # data_all = [str(path) for path in sorted(data_all, key=lambda filename: int(str(filename.name).split('_')[1]))] # for src
    # data_all = [str(path) for path in sorted(data_all, key=lambda filename: int(str(filename.name).split('_')[3][3:]))] # for tgt
    output_path = Path(args['output_path'])
    seg_out_path = output_path / 'seg'  
    vis_out_path = output_path / 'vis'  
    seg_out_path.mkdir(parents=True, exist_ok=True)
    vis_out_path.mkdir(parents=True, exist_ok=True)

    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    classifier = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False,
        progress=False,
        num_classes=args['testing_data_number_class'],
        aux_loss=None)

    checkpoint = torch.load(args['ckpt'])
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.cuda()
    classifier.eval()

    # cross_mIOU = []

    val_data = ImageDataset(data_all,
                            img_size=(args['deeplab_res'],
                                      args['deeplab_res']))
    val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

    y_preds = []
    fg_masks = []
    colorize_preds = []

    return_val = defaultdict(dict)
    with torch.no_grad():
        for _, da, in enumerate(tqdm(val_data)):
            img, img_path = da
            img_path = Path(img_path[0])

            if img.size(1) == 4:
                img = img[:, :-1, :, :]
            img = img.cuda()
            input_img_tensor = []
            for b in range(img.size(0)):
                input_img_tensor.append(resnet_transform(img[b]))
            input_img_tensor = torch.stack(input_img_tensor)

            y_pred = classifier(input_img_tensor)['out']
            y_pred = torch.log_softmax(y_pred, dim=1)
            _, y_pred = torch.max(y_pred, dim=1)
            y_pred = y_pred.permute(1,2,0).cpu().detach().numpy()

            colorize_pred = colorize_mask(y_pred[..., 0], palette)

            y_preds.append(y_pred)
            colorize_preds.append(colorize_pred)

            #             return_val[instance_id] = {}

            if save:
                img_path_root, img_name = img_path.parents[0], img_path.name
                instance_id = img_path_root.name
                img_name = img_name[:-4]  # angle id


                seg_path = output_path / 'seg' / (img_name + '.png')
                seg_img_path = output_path / 'vis' / (img_name + '.png')

                return_val[instance_id][img_name] = {}

                # np.save(seg_path, y_pred)
                # np.save(seg_path, y_pred)
                plt.imsave(seg_path, np.repeat(y_pred, 3, axis=-1).astype(np.uint8))
                plt.imsave(seg_img_path, colorize_pred)

                # fg mask, without hair and BG
                fg_mask = np.ones_like(y_pred)

                fg_mask = fg_mask[0].astype(np.uint8)  # ?
                return_val[instance_id][img_name]['ori_mask'] = fg_mask
                bg_mask = 1 - fg_mask

                fg_masks.append(fg_mask)

        return y_preds, colorize_preds, palette, fg_masks, return_val, class_name  # GUI FAN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--deeplab_res', type=int, default=512, help='output segmap resolution')
    parser.add_argument('--ckpt', type=str, default="model_dir/face_34/deeplab_class_34_checkpoint_0_filter_out_0.000000/deeplab_epoch_17.pth")
    parser.add_argument('--exp', type=str, default="experiments/face_34.json")
    parser.add_argument('--output_path', type=str, default='') 
    # parser.add_argument('--img_path', type=str)
    # parser.add_argument('--resume', type=str, default="")


    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    # class_name, palette = load_cfg(opts)
    opts.update(vars(args))

    if opts['output_path'] == '':
        dataset_path_name= Path(opts['dataset_path']).name
        opts['output_path'] = Path(opts['dataset_path']).parent / f'{dataset_path_name}_seg'

    y_preds, colorize_preds, palette, fg_masks, return_val, class_name = inference(opts, save=True)
                                                                #    ksize=11, 
                                                                #    std=0,
                                                                #    class_name_to_ignore=class_to_ignore)
