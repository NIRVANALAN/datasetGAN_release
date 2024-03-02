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
import ipdb

import pickle

face_class = ['background', 'head', 'head***cheek', 'head***chin', 'head***ear', 'head***ear***helix', 'head***ear***lobule', 'head***eye***botton lid', 'head***eye***eyelashes', 'head***eye***iris', 'head***eye***pupil', 'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid', 'head***eyebrow', 'head***forehead', 'head***frown', 'head***hair', 'head***hair***sideburns',
              'head***jaw', 'head***moustache', 'head***mouth***inferior lip', 'head***mouth***oral comisure', 'head***mouth***superior lip', 'head***mouth***teeth', 'head***neck', 'head***nose', 'head***nose***ala of nose', 'head***nose***bridge', 'head***nose***nose tip', 'head***nose***nostril', 'head***philtrum', 'head***temple', 'head***wrinkles']

eval_class_names = [
    "head***eye***iris", "head***eyebrow",
    "head***mouth***inferior lip",
    "head***mouth***superior lip",
    "head***neck",
    "head***nose",
    "head***eyebrow",
    "head***jaw",
    "head***cheek",
    "head***chin",
    "background"
] # todo, merge some classes


def merge_class(prediction, class_name): # todo, after I finished demo
    # 1 - 4
    # eye_classes = ['head***eye***botton lid', 'head***eye***eyelashes', 'head***eye***iris', 'head***eye***pupil', 'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid', 'head***eyebrow']
    # nose_classes = ['head***nose', 'head***nose***ala of nose', 'head***nose***bridge', 'head***nose***nose tip', 'head***nose***nostril']
    # mouth_classes = ['head***mouth***inferior lip', 'head***mouth***oral comisure', 'head***mouth***superior lip', 'head***mouth***teeth']
    ear_classes = ['head***ear', 'head***ear***helix', 'head***ear***lobule']
    new_class_map = {
        'eye': 1,
        'nose': 2,
        'mouth': 3,
        'ear': 4
    }
    for idx, name in enumerate(class_name):
        for current_cls_name, current_cls_idx in new_class_map.items():
            if current_cls_name in name:
                pass

def cross_validate(cp_path, args):
    if args['category'] == 'car':
        from util.data_util import car_20_palette as palette
        ignore_index = -1
    elif args['category'] == 'face':
        from util.data_util import face_palette as palette
        ignore_index = -1
    elif args['category'] == 'bedroom':
        from util.data_util import bedroom_palette as palette
        ignore_index = 0
    elif args['category'] == 'cat':
        from util.data_util import cat_palette as palette
        ignore_index = -1

    base_path = os.path.join(cp_path.split('.')[0], "cross_validation")
    # if not os.path.exists(base_path):
    Path(base_path).mkdir(parents=True, exist_ok=True)
        # os.makedirs(base_path, parents)

    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=args['testing_data_number_class'], aux_loss=None)

    cps_all = glob.glob(cp_path + "/*")

    cp_list = [data for data in cps_all if '.pth' in data and 'BEST' not in data]
    cp_list.sort()

    ids = range(args['testing_data_number_class'])

    # merge some classes
    data_all = glob.glob(args['testing_path'] + "/*")
    images = [path for path in data_all if 'npy' not in path]
    labels = [path for path in data_all if 'npy' in path]
    images.sort()
    labels.sort()

    vis_data = ImageLabelDataset(img_path_list=images,
                                  label_path_list=labels, trans=trans_mask,
                                  img_size=(args['deeplab_res'], args['deeplab_res']))
    vis_data = DataLoader(vis_data, batch_size=1, shuffle=False, num_workers=0)
    vis = []
    test_img = [] # for visuali
    for j, da, in enumerate(vis_data):
        img, mask = da[0], da[1]
        img = img.numpy()
        img = img * 255.

        img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

        mask = mask.numpy()

        curr_vis = np.concatenate( [img[0], colorize_mask(mask[0], palette)], 0 )
        test_img.append(img)
        if len(vis) < 50:

            vis.append(curr_vis)


    vis = np.concatenate(vis, 1)
    imageio.imwrite(   os.path.join(base_path, "testing.jpg"),
                      vis)
    test_data = vis.copy()

    fold_num =int( len(images) / 5)
    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])



    cross_mIOU = []
    cross_eval_mIOU = []

    for i in range(5):
        val_image = images[fold_num * i: fold_num *i + fold_num]
        val_label = labels[fold_num * i: fold_num *i + fold_num]
        test_image = [img for img in images if img not in val_image]
        test_label =[label for label in labels if label not in val_label]
        print("Val Data length,", str(len(val_image)))
        print("Testing Data length,", str(len(test_image)))

        val_data = ImageLabelDataset(img_path_list=val_image,
                                      label_path_list=val_label, trans=trans_mask,
                                      img_size=(args['deeplab_res'], args['deeplab_res']))
        val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

        test_data = ImageLabelDataset(img_path_list=test_image,
                                  label_path_list=test_label, trans=trans_mask,
                                img_size=(args['deeplab_res'], args['deeplab_res']))
        test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        best_miou = 0
        best_val_miou = 0
        # import ipdb
        # ipdb.set_trace()
        best_ckpt = None
        for resume in cp_list:

            checkpoint = torch.load(resume)
            try:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            except:
                classifier = torch.nn.DataParallel(classifier) # model .
                classifier.load_state_dict(checkpoint['model_state_dict'])


            classifier.cuda()
            classifier.eval()

            unions = {}
            intersections = {}
            for target_num in ids:
                unions[target_num] = 0
                intersections[target_num] = 0

            with torch.no_grad():
                for _, da, in enumerate(val_data):

                    img, mask = da[0], da[1]

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
                    y_pred = y_pred.cpu().detach().numpy()
                    mask = mask.cpu().detach().numpy()
                    bs = y_pred.shape[0]

                    curr_iou = []
                    if ignore_index > 0:
                        y_pred = y_pred * (mask != ignore_index)
                    for target_num in ids:
                        y_pred_tmp = (y_pred == target_num).astype(int)
                        mask_tmp = (mask == target_num).astype(int)

                        intersection = (y_pred_tmp & mask_tmp).sum()
                        union = (y_pred_tmp | mask_tmp).sum()

                        unions[target_num] += union
                        intersections[target_num] += intersection

                        if not union == 0:
                            curr_iou.append(intersection / union)
                mean_ious = []

                for target_num in ids:
                    mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))

                # save iou for each class
                # pickle.dump([intersections, unions, ids], '')

                mean_iou_val = np.array(mean_ious).mean()

                if mean_iou_val > best_val_miou:
                    best_ckpt = resume
                    best_val_miou = mean_iou_val
                    unions = {}
                    intersections = {}
                    for target_num in ids:
                        unions[target_num] = 0
                        intersections[target_num] = 0

                    with torch.no_grad():
                        testing_vis = []

    # for j, da, in enumerate(vis_data):
                        for _, da, in enumerate(vis_data):

                            img, mask = da[0], da[1]

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
                            y_pred = y_pred.cpu().detach().numpy()
                            mask = mask.cpu().detach().numpy()

                            curr_iou = []
                            if ignore_index > 0:
                                y_pred = y_pred * (mask != ignore_index)
                            for target_num in ids:
                                y_pred_tmp = (y_pred == target_num).astype(int)
                                mask_tmp = (mask == target_num).astype(int)

                                intersection = (y_pred_tmp & mask_tmp).sum()
                                union = (y_pred_tmp | mask_tmp).sum()

                                unions[target_num] += union
                                intersections[target_num] += intersection

                                if not union == 0:
                                    curr_iou.append(intersection / union)


                            img = img.cpu().numpy()
                            img =  img * 255.
                            img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

                            curr_vis = np.concatenate([img[0], colorize_mask(y_pred[0], palette)], 0)
                            if len(testing_vis) < 50:
                                testing_vis.append(curr_vis)

                        testing_vis = np.concatenate(testing_vis, 1)
                        imageio.imwrite(os.path.join(base_path, "testing_round_%d.jpg" % i),
                                          testing_vis)

                        test_mean_ious = []

                        class_wise_iou = {}
                        eval_class_iou = {}
                        eval_mean_iou = []
                        for target_num in ids:
                            class_iou = intersections[target_num] / (1e-8 + unions[target_num])
                            test_mean_ious.append(class_iou)
                            class_wise_iou[face_class[target_num]] = class_iou
                            if face_class[target_num] in eval_class_names:
                                eval_class_iou[face_class[target_num]] = class_iou
                                eval_mean_iou += [class_iou]

                            # log to local

                        best_test_miou = np.array(test_mean_ious).mean()
                        best_eval_miou = np.array(eval_mean_iou).mean()



                        print("Best IOU ,", str(best_test_miou), "CP: ", resume)
                        print("Best EVAL IOU ,", str(best_eval_miou), "CP: ", resume)

        cross_mIOU.append(best_test_miou)
        cross_eval_mIOU.append(best_eval_miou)
    
    with open(os.path.join(cp_path, 'class-iou.json'), 'w') as f:
        json.dump(class_wise_iou, f) 
    with open(os.path.join(cp_path, 'class-eval-iou.json'), 'w') as f:
        json.dump(eval_class_iou, f) 

    print(cross_mIOU)
    print(" cross validation mean:" , np.mean(cross_mIOU) )
    print(" cross validation std:", np.std(cross_mIOU))

    print(cross_eval_mIOU)
    print(" cross EVAL validation mean:" , np.mean(cross_eval_mIOU) )
    print(" cross EVAL validation std:", np.std(cross_eval_mIOU))

    result = {"Cross validation mean": np.mean(cross_mIOU), "Cross validation std": np.std(cross_mIOU), "Cross validation":cross_mIOU }
    with open(os.path.join(cp_path, 'cross.json'), 'w') as f:
        json.dump(result, f)

def test(cp_path, args, validation_number=50):
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

    ipdb.set_trace()


    cps_all = glob.glob(cp_path + "/*")

    cp_list = [data for data in cps_all if '.pth' in data and 'BEST' not in data]

    ids = range(args['testing_data_number_class'])

    data_all = glob.glob(args['testing_path'] + "/*")
    images = [path for path in data_all if 'npy' not in path]
    labels = [path for path in data_all if 'npy' in path]
    images.sort()
    labels.sort()

    vis_data = ImageLabelDataset(img_path_list=images,
                                  label_path_list=labels, trans=trans_mask,
                                  img_size=(args['deeplab_res'], args['deeplab_res']))
    vis_data = DataLoader(vis_data, batch_size=1, shuffle=False, num_workers=0)
    vis = []
    for j, da, in enumerate(vis_data):
        img, mask = da[0], da[1]
        img = img.numpy()
        img = img * 255.

        img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

        mask = mask.numpy()

        curr_vis = np.concatenate( [img[0], colorize_mask(mask[0], palette)], 0 )
        if len(vis) < 50:

            vis.append(curr_vis)

    vis = np.concatenate(vis, 1)
    imageio.imwrite(   os.path.join(base_path, "testing_gt.jpg"),
                      vis)


    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=args['testing_data_number_class'], aux_loss=None)

    cross_mIOU = []
    if validation_number == 0:
        print("Report performance on the best checkpoint")
        val_image = images
        val_label = labels
        test_image = images
        test_label = labels
    else:
        val_image = images[:validation_number]
        val_label = labels[:validation_number]
        test_image = [img for img in images if img not in val_image]
        test_label =[label for label in labels if label not in val_label]
    print("Val Data length,", str(len(val_image)))
    print("Testing Data length,", str(len(test_image)))

    val_data = ImageLabelDataset(img_path_list=val_image,
                                  label_path_list=val_label, trans=trans_mask,
                                  img_size=(args['deeplab_res'], args['deeplab_res']))
    val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    test_data = ImageLabelDataset(img_path_list=test_image,
                              label_path_list=test_label, trans=trans_mask,
                            img_size=(args['deeplab_res'], args['deeplab_res']))
    test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    best_val_miou = 0

    cp_list.sort()
    best_ckpt = None
    for resume in cp_list:
        checkpoint = torch.load(resume)
        classifier.load_state_dict(checkpoint['model_state_dict'])


        classifier.cuda()
        classifier.eval()

        unions = {}
        intersections = {}
        for target_num in ids:
            unions[target_num] = 0
            intersections[target_num] = 0

        with torch.no_grad():
            for _, da, in enumerate(val_data):

                img, mask = da[0], da[1]

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
                y_pred = y_pred.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                bs = y_pred.shape[0]

                curr_iou = []

                for target_num in ids:
                    y_pred_tmp = (y_pred == target_num).astype(int)
                    mask_tmp = (mask == target_num).astype(int)

                    intersection = (y_pred_tmp & mask_tmp).sum()
                    union = (y_pred_tmp | mask_tmp).sum()

                    unions[target_num] += union
                    intersections[target_num] += intersection

                    if not union == 0:
                        curr_iou.append(intersection / union)
            mean_ious = []

            for target_num in ids:
                mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
            mean_iou_val = np.array(mean_ious).mean()

            if mean_iou_val > best_val_miou:
                best_val_miou = mean_iou_val
                best_ckpt = resume
                unions = {}
                intersections = {}
                for target_num in ids:
                    unions[target_num] = 0
                    intersections[target_num] = 0

                with torch.no_grad():
                    testing_vis = []
                    for _, da, in enumerate(test_data):

                        img, mask = da[0], da[1]

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
                        y_pred = y_pred.cpu().detach().numpy()
                        mask = mask.cpu().detach().numpy()

                        curr_iou = []

                        for target_num in ids:
                            y_pred_tmp = (y_pred == target_num).astype(int)
                            mask_tmp = (mask == target_num).astype(int)

                            intersection = (y_pred_tmp & mask_tmp).sum()
                            union = (y_pred_tmp | mask_tmp).sum()

                            unions[target_num] += union
                            intersections[target_num] += intersection

                            if not union == 0:
                                curr_iou.append(intersection / union)


                        img = img.cpu().numpy()
                        img =  img * 255.
                        img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

                        curr_vis = np.concatenate([img[0], colorize_mask(y_pred[0], palette)], 0)
                        if len(testing_vis) < 50:
                            testing_vis.append(curr_vis)

                    testing_vis = np.concatenate(testing_vis, 1)
                    imageio.imsave(os.path.join(base_path, "testing.jpg"),
                                      testing_vis)

                    test_mean_ious = []

                    for j, target_num in enumerate(ids):
                        iou = intersections[target_num] / (1e-8 + unions[target_num])
                        print("IOU for ", class_name[j], iou)

                        test_mean_ious.append(iou)
                    best_test_miou = np.array(test_mean_ious).mean()
                    print("Best IOU ,", str(best_test_miou), "CP: ", resume)

    print(cross_mIOU)
    print("Validation mIOU:" ,best_val_miou)
    print("Testing mIOU:" , best_test_miou )

    result = {"Validation": best_val_miou, "Testing":best_test_miou}
    with open(os.path.join(cp_path, 'cross.json'), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--cross_validate', type=bool, default=False)

    parser.add_argument('--validation_number', type=int, default=0)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))


    # ipdb.set_trace()
    if not args.cross_validate:
        test(args.resume, opts, args.validation_number)
    else:
        cross_validate(args.resume, opts)

