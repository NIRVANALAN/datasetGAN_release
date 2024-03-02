import torch
from pdb import set_trace as st
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms.functional import InterpolationMode
import random
import ipdb
import torchvision.datasets as dsets
from torchvision import transforms
from PIL import Image
import os
from .data_transforms import *
from pathlib import Path

from .seg_utils import generate_label_argmax

celebamask_class_name = {
    0: "background",
    1: "skin",
    2: "nose",
    3: "eye_g",
    4: "l_eye",
    5: "r_eye",
    6: "r_brow",
    7: "l_brow",
    8: "r_ear",
    9: "l_ear",
    10: "mouth",
    11: "u_lip",
    12: "l_lip",
    13: "hair",
    14: "hat",
    15: "ear_r",
    16: "neck_l",
    17: "neck",
    18: "cloth",
}

celebamask_value_key_reverse = {v: k for k, v in celebamask_class_name.items()}


class ImageLabelDataset(Dataset):

    def __init__(
            self,
            # img_path_list,
            # label_path_list,
            img_size=(128, 128),
    ):

        images = []
        labels = []
        data_path = '/mnt/lustre/yslan/Dataset/CVPR22/seg'
        dataset_root = list(Path(data_path).glob('*seed*'))
        for sub_dataset in dataset_root:
            for sub_dir in sub_dataset.iterdir():  # psi
                for sample in sub_dir.iterdir():
                    num_imgs = len(list(sample.glob('0_*.png')))
                    for idx in range(num_imgs):
                        img_name = sample / f'0_0.5_random_angle{idx}.png'
                        label_name = sample / f'{idx}.npy'
                        if not label_name.exists():
                            label_name = sample / f'{idx}.png'

                        if not (img_name.exists() and label_name.exists()):
                            continue
                        # assert img_name.exists(), img_name
                        # assert label_name.exists(), label_name

                        images.append(str(img_name))
                        labels.append(str(label_name))

        self.img_path_list = images
        self.label_path_list = labels
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
                lbl = np.load(lbl_path)  # npy
            elif suffix == '.png':
                lbl = np.array(Image.open(lbl_path))  # png
        except:
            print(lbl_path)

        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]

        # lbl = self.label_trans(lbl)
        lbl = Image.fromarray(lbl.astype('uint8'))
        im, lbl = self.transform(im, lbl)

        return im, lbl, im_path, lbl_path

    def transform(self, img, lbl):
        img = img.resize((self.img_size, self.img_size))
        lbl = lbl.resize((self.img_size, self.img_size),
                         resample=Image.NEAREST)
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


class CelebAMaskHQ:

    def __init__(
        self,
        config,
        img_path,
        label_path,
        transform_img,
        transform_label,
        mode,
        train_split_size,
        suffix="jpg",
    ):
        self.img_path = img_path
        self.config = config
        self.label_path = label_path
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.train_dataset = []
        self.test_dataset = []
        self.mode = mode
        self.train_split_size = train_split_size
        self.img_suffix = suffix
        self.aug = config.aug  # needs to apply the same operation of img and label
        self.preprocess()

        if mode == True:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

        # prepare imgsA
        # self.imgsA = self._read_file(12)

    def preprocess(self):

        for i in range(
                len([
                    name for name in os.listdir(self.img_path)
                    if os.path.isfile(os.path.join(self.img_path, name))
                ])):
            # img_path = os.path.join(self.img_path, str(i)+'.jpg')
            img_path = os.path.join(self.img_path,
                                    str(i) + f".{self.img_suffix}")

            label_path = os.path.join(self.label_path, str(i) + ".png")
            # print (img_path, label_path)
            if self.mode == True:
                self.train_dataset.append([img_path, label_path])
                # print(f'Training dataset size: {len(self.train_dataset)}')
            else:
                self.test_dataset.append([img_path, label_path])
                # print(f'Eval dataset size: {len(self.test_dataset)}')

        if self.config.shuffle:
            random.shuffle(self.train_dataset)
            print("shuffle train_dataset")

        if self.train_split_size != -1:
            self.train_dataset = self.train_dataset[:self.train_split_size]

        print(
            f"Finished preprocessing the CelebA dataset: {len(self.train_dataset) if self.mode else len(self.test_dataset)}"
        )


    def _map_seg_label_coordgan(self, seg_label):
        # follow https://github.com/NVlabs/CoordGAN/blob/b18cb817f16b86bb1e5899bd5ee59ab62de845d0/seg_utils.py
        celebamaskhq_filted_parts = ['skin', 'u_lip', 'l_lip', 'l_eye', 'r_eye', 'l_brow', 'r_brow', \
    'l_ear', 'r_ear', 'nose',  'mouth']

        coordgan_lbls = {}
        coordgan_lbls['u_lip'] = 1
        coordgan_lbls['l_lip'] = 1
        coordgan_lbls['mouth'] = 1
        coordgan_lbls['l_eye'] = 2
        coordgan_lbls['r_eye'] = 2
        coordgan_lbls['l_brow'] = 3
        coordgan_lbls['r_brow'] = 3
        coordgan_lbls['l_ear'] = 4
        coordgan_lbls['r_ear'] = 4
        coordgan_lbls['nose'] = 5
        coordgan_lbls['skin'] = 6

        seg_img = np.zeros_like(seg_label)

        for idx in range(len(celebamaskhq_filted_parts)):
            part_name = celebamaskhq_filted_parts[idx]
            seg_part_index = celebamask_value_key_reverse[part_name]
            seg_label_part_indices = seg_label == seg_part_index
            seg_img[seg_label_part_indices] = coordgan_lbls[part_name]

        return torch.from_numpy(seg_img)

    def _read_file(self, index):

        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)
        # ipdb.set_trace()

        if self.aug:
            state = torch.get_rng_state()
            image = self.transform_img(image)
            torch.set_rng_state(state)
            label = self.transform_label(label)
        else:
            image, label = self.transform_img(image), self.transform_label(
                label)
            
        # * image [-1,1]  to [0,1] for dsgan input
        image = image / 2 + 0.5

        label = 255 * np.array(label)
        # label = self._map_seg_label_coordgan(label).int()

        # ! for debugging
        # fg_mask = label!=0
        # label[label!=1] = 0 # mouth, bug in the data? 
        # label[label!=2] = 0 # mouth, bug in the data? 
        # label[label!=5] = 0 # nose, fails why?
        # label *= fg_mask

        # st()

        # i, j = torch.where(label[0] > 0)
        # points = torch.stack([j, i], -1)  # N, 2
        # alpha_channels = torch.ones_like(points)[:, 0].float()  # N, 1

        # label_colors = (label[:, i, j]).squeeze().reshape(1, 1, -1,
        #                                                   1).float()  # 1 C H W

        # # * seg label apping

        # label_colors_colorize = generate_label_argmax(
        #     label_colors, n_label=7).squeeze().permute(1, 0).float()  # N 3
        
        # seg_label_colors = label_colors.reshape(-1)

        # kpts = torch.nonzero(torch.ones_like(label))[:, 1:] # H*W, 2
        # * debug on "nose" class now
        # kpts = torch.nonzero(label)[:, 1:]

        # return image, label, img_path, label_path
        # return {

            # "imgsA": self.imgsA['imgsB'],
            # "kptsA": self.imgsA['kptsB'],
            # "labelA": self.imgsA['labelB'],

            # "imgsA_path": self.imgsA['imgsB_path'],
            # "labelsA_path": self.imgsA['labelsB_path'],
            # "imgsB": image,
            # "labelB": (label * 255).float(),
            # "imgsB_path": img_path,
            # "labelsB_path": label_path,
            # "pointsB": points,
            # 'label_colors': label_colors_colorize,
            # 'label_pointsB': seg_label_colors,
            # 'alpha_channels': alpha_channels,
        # }

        # st()

        return image, label

    def __getitem__(self, index):

        # dataset = self.train_dataset if self.mode == True else self.test_dataset
        # img_path, label_path = dataset[index]
        # image = Image.open(img_path)
        # label = Image.open(label_path)
        # # ipdb.set_trace()

        # if self.aug:
        #     state = torch.get_rng_state()
        #     image = self.transform_img(image)
        #     torch.set_rng_state(state)
        #     label = self.transform_label(label)
        # else:
        #     image, label = self.transform_img(image), self.transform_label(label)

        # get the foreground indices for segmentationt ransfer
        # imgsB = self._read_file(index)
        return self._read_file(index)
        # return {
        #     **imgsB,
        #     "imgsA": self.imgsA['imgsB'],
        #     "pointsA": self.imgsA['pointsB'],
        #     "labelA": self.imgsA['labelB'],
        #     "imgsA_path": self.imgsA['imgsB_path'],
        #     "labelsA_path": self.imgsA['labelsB_path'],
        #     'label_colors_A': self.imgsA['label_colors'],
        #     'alpha_channels_A': self.imgsA['alpha_channels'],
        #     'label_pointsA': self.imgsA['label_pointsB'],
        # }

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class CelebAMaskHQ_lms(CelebAMaskHQ):

    def __init__(self,
                 config,
                 img_path,
                 label_path,
                 transform_img,
                 transform_label,
                 mode,
                 train_split_size,
                 suffix="jpg"):
        super().__init__(config, img_path, label_path, transform_img,
                         transform_label, mode, train_split_size, suffix)
        lms_path = '/mnt/lustre/yslan/Repo/Research/ijcv-2023/ddf-eg3d-e3dge/eg3d/out/template/lms64.npy'
        template_img_path = '/mnt/lustre/yslan/Repo/Research/ijcv-2023/ddf-eg3d-e3dge/eg3d/out/template/seed0000.png'

        self.lms = np.load(lms_path) / 2  # normalize to 256
        self.lms_template_img = self.transform_img(
            Image.open(template_img_path))

    def _read_file(self, index):

        dataset = self.train_dataset if self.mode == True else self.test_dataset
        img_path, label_path = dataset[index]
        image = Image.open(img_path)
        label = Image.open(label_path)

        if self.aug:
            state = torch.get_rng_state()
            image = self.transform_img(image)
            torch.set_rng_state(state)
            label = self.transform_label(label)
        else:
            image, label = self.transform_img(image), self.transform_label(
                label)

        # kpts = torch.nonzero(torch.ones_like(label))[:, :2] # H*W, 2
        # * debug on "nose" class now
        # kpts = torch.nonzero(label*255==2)[:, 1:]

        # return image, label, img_path, label_path
        return {

            # "imgsA": self.imgsA['imgsB'],
            # "kptsA": self.imgsA['kptsB'],
            # "labelA": self.imgsA['labelB'],

            # "imgsA_path": self.imgsA['imgsB_path'],
            # "labelsA_path": self.imgsA['labelsB_path'],
            "imgsB": image,
            # "kptsB": kpts,
            "labelB": label,
            "imgsB_path": img_path,
            "labelsB_path": label_path,
        }

    def __getitem__(self, index):

        # dataset = self.train_dataset if self.mode == True else self.test_dataset
        # img_path, label_path = dataset[index]
        # image = Image.open(img_path)
        # label = Image.open(label_path)
        # # ipdb.set_trace()

        # if self.aug:
        #     state = torch.get_rng_state()
        #     image = self.transform_img(image)
        #     torch.set_rng_state(state)
        #     label = self.transform_label(label)
        # else:
        #     image, label = self.transform_img(image), self.transform_label(label)

        # get the foreground indices for segmentationt ransfer
        imgsB = self._read_file(index)
        return {
            **imgsB,
            "imgsA": self.lms_template_img,
            "kptsA": self.lms,
            # "labelA": self.imgsA['labelB'],
            # "imgsA_path": self.imgsA['imgsB_path'],
            # "labelsA_path": self.imgsA['labelsB_path'],
        }


class Data_Loader:

    def __init__(
        self,
        config,
        img_path,
        label_path,
        image_size,
        batch_size,
        mode,
        train_split_size=-1,
        suffix=".jpg",
        load_lms=False,
    ):
        self.img_path = img_path
        self.config = config
        self.label_path = label_path
        self.imsize = image_size
        self.batch = batch_size
        self.mode = mode
        self.img_suffix = suffix
        self.train_split_size = train_split_size
        self.load_lms = load_lms

    def transform_img(
        self,
        resize,
        totensor,
        normalize,
        centercrop,
        randomflip=False,
        randomaffine=False,
        perspective=False,
        jitter=False,
    ):
        options = []

        if randomflip:
            options.append(transforms.RandomHorizontalFlip(0.5))
        if randomaffine:
            options.append(transforms.RandomAffine(45))
        if perspective:
            options.append(transforms.RandomPerspective(0.2))
        if jitter:
            options.append(RandomJitter(0.4, 0.4, 0.4))

        if centercrop:
            options.append(transforms.CenterCrop(160))

        if resize:
            if self.config.downsample_size != -1:
                options.append(
                    transforms.Resize((self.config.downsample_size,
                                       self.config.downsample_size)))

            options.append(transforms.Resize((self.imsize, self.imsize)))

        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)

        print(f'Train Mode: {self.mode}, image data_pipeline: {transform}')
        return transform

    def transform_label(
        self,
        resize,
        totensor,
        normalize,
        centercrop,
        randomflip=False,
        randomaffine=False,
        perspective=False,
    ):
        options = []
        if randomflip:
            options.append(transforms.RandomHorizontalFlip(0.5))
        if randomaffine:
            options.append(transforms.RandomAffine(45))
        if perspective:
            options.append(transforms.RandomPerspective(0.2))

        if centercrop:
            options.append(transforms.CenterCrop(160))

        if resize:
            # if self.config.downsample_size!=-1:
            #     options.append(transforms.Resize((self.config.downsample_size, self.config.downsample_size), interpolation=0))

            # options.append(transforms.Resize((self.imsize, self.imsize), interpolation=0))
            options.append(transforms.Resize((self.imsize, self.imsize)))

        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0, 0, 0), (0, 0, 0)))
        transform = transforms.Compose(options)
        print(f'Train Mode: {self.mode}, label data_pipeline: {transform}')
        return transform

    def loader(self):
        if self.config.aug and self.mode:  # Open aug in train mode
            data_pipeline = dict(
                resize=True,
                totensor=True,
                normalize=True,
                centercrop=False,
                randomflip=self.config.flip,
                randomaffine=self.config.affine,
                perspective=self.config.perspective,
                jitter=self.config.jitter,
            )
            test_pipeline = data_pipeline.copy()
            test_pipeline.pop("jitter")
            test_pipeline.update(dict(normalize=False))
            # print(f"data pipeline: {data_pipeline}")
            # print(f"test pipeline: {test_pipeline}")
            transform_img = self.transform_img(
                #                True, True, True, False, True, True, True
                **data_pipeline)
            transform_label = self.transform_label(
                #                True, True, False, False, True, True, True
                **test_pipeline)
            # * apply the same operation
            print("DA Enabled")
        else:
            transform_img = self.transform_img(True, True, True, False)
            transform_label = self.transform_label(True, True, False, False)

        print()

        if self.load_lms:

            dataset = CelebAMaskHQ_lms(
                self.config,
                self.img_path,
                self.label_path,
                transform_img,
                transform_label,
                self.mode,
                self.train_split_size,
                self.img_suffix,
            )

        else:

            dataset = CelebAMaskHQ(
                self.config,
                self.img_path,
                self.label_path,
                transform_img,
                transform_label,
                self.mode,
                self.train_split_size,
                self.img_suffix,
            )

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=self.mode,
            num_workers=1,
            # num_workers=0,
            drop_last=True,
        )
        return loader
