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

from ipdb import set_trace as st
import ipdb
import sys
sys.path.append('../')
import os
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

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]
        im = Image.open(im_path)

        suffix = Path(lbl_path).suffix
        
        try:
            if suffix == '.npy':
                lbl = np.load(lbl_path) # npy
            elif suffix == '.png':
                lbl = np.array(Image.open(lbl_path)) # png
        except:
            print(lbl_path)

        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]

        lbl = self.label_trans(lbl)
        lbl = Image.fromarray(lbl.astype('uint8'))
        im, lbl = self.transform(im, lbl)

        return im, lbl, im_path

    def transform(self, img, lbl):
        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = transforms.ToTensor()(img)
        return img, lbl


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

def main(data_path, args, resume, max_data=0, uncertainty_portion=0, ddf_sample=False):
    exp_path = args['exp_dir']

    base_path = os.path.join(exp_path, "deeplab_class_%d_checkpoint_%d_filter_out_%f" %(args['testing_data_number_class'],
                                                                                        int(max_data),
                                                                                        uncertainty_portion))
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    print("Model dir,", base_path)
    num_class = args['testing_data_number_class']

    if not ddf_sample:

        dump_data = []
        all_pickle = glob.glob(data_path + '/*.pickle')

        used_image = []
        for p in all_pickle:
            with open(p, 'rb') as f:
                curr_dict = pickle.load(f)

            for dd in curr_dict:
                if not dd['image_name'] in used_image:
                    used_image.append(dd['image_name'] )
                    dump_data.append(dd)
        if max_data > 0:
            dump_data = dump_data[:max_data]

        # ipdb.set_trace()

        stylegan_images = [data['image_name'] for data in dump_data]
        stylegan_labels = [data['image_label_name'] for data in dump_data]

        stylegan_images.sort()
        stylegan_labels.sort()
        if uncertainty_portion > 0:
            sort_by_uncertainty = sorted(dump_data, key=lambda k: k['uncertrainty_score'])
            filter_out_num = int(len(sort_by_uncertainty) * uncertainty_portion)
            sort_by_uncertainty = sort_by_uncertainty[30:-filter_out_num+ 30]
            out_idx = range(len(sort_by_uncertainty))
            stylegan_images = [sort_by_uncertainty[idx]['image_name'] for idx in out_idx]
            stylegan_labels = [sort_by_uncertainty[idx]['image_label_name'] for idx in out_idx]
        stylegan_images = [data['image_name'] for data in dump_data]
        stylegan_labels = [data['image_label_name'] for data in dump_data]

    else:
        # stylegan_images = []
        # stylegan_labels = []
        # dataset_root = list(Path(data_path).glob('*seed*'))
        # for sub_dataset in dataset_root:
        #     for sub_dir in sub_dataset.iterdir(): # psi
        #         for sample in sub_dir.iterdir():
        #             num_imgs = len(list(sample.glob('0_*.png')))
        #             for idx in range(num_imgs):
        #                 img_name = sample / f'0_0.5_random_angle{idx}.png'
        #                 label_name = sample / f'{idx}.npy' 
        #                 if not label_name.exists():
        #                     label_name = sample / f'{idx}.png' 

        #                 if not (img_name.exists() and label_name.exists()):
        #                     continue
        #                 # assert img_name.exists(), img_name
        #                 # assert label_name.exists(), label_name

        #                 stylegan_images.append(str(img_name))
        #                 stylegan_labels.append(str(label_name))

        # update 2022 Sept 8
        stylegan_images = []
        stylegan_labels = []
        dataset_root = list(Path(data_path).glob('*seed*'))
        if len(dataset_root) == 0:
            dataset_root = [Path(data_path)]
        for sub_dataset in dataset_root:

            img_dir = sub_dataset / 'for_dsgan_seg'
            label_dir = sub_dataset / 'for_dsgan_seg_label'

            num_imgs = len(list(img_dir.glob('*.png')))

            for idx in range(num_imgs):
                img_name = img_dir / f'{idx:04}.png'   # ffhq
                label_name = label_dir / f'{idx}.png'  

                if not (img_name.exists() and label_name.exists()):
                    continue

                stylegan_images.append(str(img_name))
                stylegan_labels.append(str(label_name))



    if args['number_class'] ==  args['testing_data_number_class']:
        trans_method = trans_mask
    else:
        # for testing on ADE-12 only. Since our generated images has 20 labels. Need to merge labels based on testing set.
        trans_method = trans_mask_stylegan_20classTo12
    assert  len(stylegan_images) == len(stylegan_labels)
    print( "Train data length,", str(len(stylegan_labels)))

    train_data = ImageLabelDataset(img_path_list=stylegan_images,
                              label_path_list=stylegan_labels, trans=trans_method,
                            img_size=(args['deeplab_res'], args['deeplab_res']), )

    train_data = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True, num_workers=16, drop_last=True)
    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=num_class, aux_loss=None)
    if resume != "":
        checkpoint = torch.load(resume)
        classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier.cuda()
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    if num_gpus > 1:
        parallel_flag = True
        classifier = nn.DataParallel(classifier)
        print('traing parallel dp')
    else:
        parallel_flag = False
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


    for epoch in range(20):
        for i, da, in enumerate(train_data):
            # continue # dry run

            # if da[0].shape[0] != 8: # ?????
            #     continue

            if i % 10 == 0:
                gc.collect()

            classifier.train()

            optimizer.zero_grad()
            img, mask = da[0], da[1]

            img = img.cuda()
            mask = mask.cuda()

            input_img_tensor = []
            for b in range(img.size(0)):
                if img.size(1) == 4:
                    input_img_tensor.append(resnet_transform(img[b][:-1,:,:]))
                else:
                    input_img_tensor.append(resnet_transform(img[b]))

            input_img_tensor = torch.stack(input_img_tensor)

            y_pred = classifier(input_img_tensor)['out']
            loss = criterion(y_pred, mask)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(epoch, 'epoch', 'iteration', i, 'loss', loss.item(), flush=True)

        model_path = os.path.join(base_path, 'deeplab_epoch_' + str(epoch) + '.pth')

        print('Save to:', model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str,  default="")

    parser.add_argument('--max_data', type=int,  default=0)
    parser.add_argument('--uncertainty_portion', type=float,  default=0)
    parser.add_argument('--ddf_sample', action='store_true')


    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    main(args.data_path, opts, args.resume, args.max_data, args.uncertainty_portion, args.ddf_sample)


