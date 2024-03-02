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
import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import gc
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import glob
from util.data_util import *
from util.utils import process_image, colorize_mask
import json
from train_deeplab import ImageLabelDataset
import scipy.misc
from pathlib import Path
import imageio

from pdb import set_trace as st

from pathlib import Path

from datasets import seg_dataloader

from tqdm import tqdm

def str2bool(v):
    return v.lower() in ('true')

def test(cp_path, args, parsed_args, validation_number=50):
    if args['category'] == 'car':
        from util.data_util import car_20_palette as palette
        if args['testing_data_number_class'] == 12:
            from util.data_util import car_12_class as class_name
        elif args['testing_data_number_class'] == 20:
            from util.data_util import car_20_class as class_name
    elif args['category'] == 'face':
        from util.data_util import face_palette as palette
        from util.data_util import face_class as class_name

    elif args['category'] == 'bedroom':
        from util.data_util import bedroom_palette as palette
    elif args['category'] == 'cat':
        from util.data_util import cat_palette as palette

    base_path = os.path.join(cp_path, "validation")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=args['testing_data_number_class'], aux_loss=None)

    val_data = seg_dataloader(parsed_args, load_lms=False) 
    checkpoint = torch.load(parsed_args.resume)

    classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier.cuda()
    classifier.eval()

    with torch.no_grad():
        for batch_idx, da, in enumerate(tqdm(val_data)):

            img, mask = da[0], da[1]

            if mask.ndim == 4:
                mask = mask[:, 0, ...]

            if img.size(1) == 4:
                img = img[:, :-1, :, :]

            img = img.cuda()
            mask = mask.cuda()

            input_img_tensor = []
            for b in range(img.size(0)):
                input_img_tensor.append(resnet_transform(img[b]))

            input_img_tensor = torch.stack(input_img_tensor)
        

            y_pred = classifier(input_img_tensor)['out']
            y_pred = torch.log_softmax(y_pred, dim=1)
            _, y_pred = torch.max(y_pred, dim=1)
            y_pred_mask = y_pred.cpu().detach().numpy().astype(np.uint8)
            
            img_vis = (img.cpu().numpy() / 2 + 0.5) * 255

            img_vis = np.transpose(img_vis, (0, 2, 3, 1)).astype(np.uint8)

            mask_gt = mask.cpu().numpy()

            curr_vis = np.concatenate( [img_vis[0], colorize_mask(y_pred_mask[0], palette).astype(np.uint8), img_vis[0].astype(np.uint8) * 0.5 + 0.5 * colorize_mask(mask_gt[0], palette).astype(np.uint8)], 1 ) # concat in H

            # testing_vis = np.concatenate(testing_vis, 1)
            imageio.imsave(os.path.join(base_path, "vis_{}.png".format(batch_idx)), curr_vis)
            imageio.imsave(os.path.join(base_path, "{}.png".format(batch_idx)), y_pred_mask[0])
            # imageio.imsave(os.path.join(base_path, "{}.png".format(batch_idx)), y_pred_mask[0])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--cross_validate', type=bool, default=False)
    parser.add_argument('--validation_number', type=int, default=0)

    # ========= copied from CelebAMask ============

    # Model hyper-parameters
    parser.add_argument('--model',
                        type=str,
                        default='parsenet',
                        choices=['parsenet'])
    parser.add_argument('--downsample_size', type=int, default=-1)
    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument('--version', type=str, default='parsenet')

    # Training setting
    parser.add_argument('--total_step',
                        type=int,
                        default=100000,
                        help='how many times to update the generator')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--train_split_size',
                        type=int,
                        default=-1,
                        help='how many times to update the generator')

    # Testing setting
    parser.add_argument('--test_size', type=int, default=2824)
    parser.add_argument('--model_name', type=str, default='model.pth')

    # using pretrained
    # parser.add_argument('--pretrained_model', type=int, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--aug', type=str2bool, default=False)
    # DA
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--affine', action='store_true')
    parser.add_argument('--perspective', action='store_true')
    parser.add_argument('--jitter', action='store_true')
    parser.add_argument('--same_res', action='store_true')

    # Path
    parser.add_argument('--img_path',
                        type=str,
                        default='./Data_preprocessing/train_img')
    parser.add_argument('--label_path',
                        type=str,
                        default='./Data_preprocessing/train_label')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--test_image_path',
                        type=str,
                        default='./Data_preprocessing/test_img')
    #parser.add_argument('--test_label_path', type=str, default='./test_results')  # ?????
    parser.add_argument('--test_label_path',
                        type=str,
                        default='./Data_preprocessing/test_label')  # ????????
    parser.add_argument('--test_color_label_path',
                        type=str,
                        default='./test_color_visualize')
    parser.add_argument('--img_suffix', type=str, default='jpg')

    # log
    parser.add_argument('--test_label_save_path',
                        type=str,
                        default='./test_results')
    parser.add_argument('--test_color_label_save_path',
                        type=str,
                        default='./test_color_visualize')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)
    parser.add_argument('--eval_step', type=int, default=2000)


    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    test(str(Path(args.resume).parent), opts, args)

