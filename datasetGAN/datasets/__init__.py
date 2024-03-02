from torch.utils import data
from torchvision import transforms
from datasets.dataset import MultiResolutionDataset, sample_infinite_data
from datasets.pck_dataset import PCKDataset, sample_infinite_pck_data

from .seg_dataloader import Data_Loader as seg_DataLoader


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])


# Convenience functions to easily load data:
def img_dataloader(path=None, transform=_transform, resolution=256, seed=0, batch_size=64, shuffle=True, distributed=True,
                   dset=None, return_indices=False, infinite=True, subset=None, drop_last=True):
    dset = MultiResolutionDataset(path, transform, resolution, return_indices) if dset is None else dset
    if subset is not None:
        dset = data.Subset(dset, subset)
    loader = data.DataLoader(dset, batch_size=batch_size,
                             sampler=data_sampler(dset, shuffle=shuffle, distributed=distributed),
                             drop_last=drop_last)
    if infinite:
        loader = sample_infinite_data(loader, seed)
    return loader


def pck_dataloader(path, transform=_transform, resolution=256, seed=0, batch_size=64, distributed=True, infinite=True):
    dset = PCKDataset(path, transform, resolution, seed)
    # shuffling is handled internally by PCKDataset when infinite=True:
    loader = data.DataLoader(dset, batch_size=batch_size,
                             sampler=data_sampler(dset, shuffle=False, distributed=distributed),
                             drop_last=False)
    if infinite:
        loader = sample_infinite_pck_data(loader, seed)
    return loader

def seg_dataloader(config, load_lms=False):
    
    eval_loader = seg_DataLoader(
    config,
    config.test_image_path,
    config.test_label_path,
    config.imsize,
    config.batch_size,
    # config.train,
    False,
    config.test_size,
    # 'jpg', # test, jpg; train png
    'png', # test, jpg; train png
    load_lms=load_lms,
    ).loader()

    return eval_loader
