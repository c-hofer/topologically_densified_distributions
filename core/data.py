import torchvision
import torch
import os
import numpy as np
import pickle


from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.dataset import Subset, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST
from .ds_util import Transformer
from pathlib import Path


def CIFAR_1_1(root):
    root = Path(root)
    X = np.load(root / 'datasets/cifar10.1_v4_data.npy')
    X = torch.from_numpy(X).float().permute(0, 3, 1, 2)/256.
    y = np.load(root / 'datasets/cifar10.1_v4_labels.npy')
    y = torch.from_numpy(y).long()
    ds = TensorDataset(X, y)
    ds = Transformer(ds, transform=torchvision.transforms.ToPILImage())
    return ds


class DatasetFactory():
    def __init__(self,
                 ds_name_to_fn_and_kwargs
                 ):

        self.ds_name_to_fn_and_kwargs = ds_name_to_fn_and_kwargs

    def __call__(self,
                 ds_name: str):

        ds_name = str(ds_name)
        assert ds_name in self.ds_name_to_fn_and_kwargs

        try:
            ds_fn, kwargs = self.ds_name_to_fn_and_kwargs[ds_name]

        except KeyError:
            raise KeyError('Expected ds_name to be in {} but got {}'.format(
                list(self.ds_name_to_fn_and_kwargs.keys()),
                ds_name)
            )

        return ds_fn(**kwargs)

    def __iter__(self):
        return iter(self.ds_name_to_fn_and_kwargs)


DATA_ROOT = Path('/scratch1/chofer/data/')

DS_PATH_CFG = {
    'cifar10_train':
        (CIFAR10, {'root': DATA_ROOT / 'cifar10', 'train': True}),
    'cifar10_test':
        (CIFAR10, {'root': DATA_ROOT / 'cifar10', 'train': False}),
    'cifar10.1v4':
        (CIFAR_1_1, {'root': DATA_ROOT / 'CIFAR-10.1'}),
    'cifar100_train':
        (CIFAR100, {'root': DATA_ROOT / 'cifar100', 'train': True}),
    'cifar100_test':
        (CIFAR100, {'root': DATA_ROOT / 'cifar100', 'train': False}),
    'SVHN_train':
        (SVHN, {'root': DATA_ROOT / 'SVHN', 'split': 'train'}),
    'SVHN_test':
        (SVHN, {'root': DATA_ROOT / 'SVHN', 'split': 'test'}),
    'MNIST_train':
        (MNIST, {'root': DATA_ROOT / 'mnist', 'train' : True, 'download': True}),
    'MNIST_test':
        (MNIST, {'root': DATA_ROOT / 'mnist', 'train' : False, 'download' : True})

}

SPLIT_INDICES_PTH = Path(Path(__file__).parent) / 'data_train_indices.pickle'

ds_factory = DatasetFactory(DS_PATH_CFG)


DS_SPLIT_CFG = {
    'cifar10_train': [100, 250, 500, 1000, 2000, 4000],
    'cifar100_train': [100, 250, 500, 1000, 2000, 2500, 4000, 10000],
    'SVHN_train': [100, 250, 500, 1000, 2000, 4000],
    'MNIST_train': [100, 250, 500, 1000, 2000, 4000]
}
DS_SPLIT_CFG_NUM_SPLITS = 20


def verify_split_cache():
    if SPLIT_INDICES_PTH.is_file():
        with open(SPLIT_INDICES_PTH, 'br') as fid:
            cache = pickle.load(fid)

    else:
        cache = {}

    for ds_name, needed_num_samples in DS_SPLIT_CFG.items():
        for num_samples in needed_num_samples:
            if (ds_name, num_samples) in cache:
                continue
            else:
                ds = ds_factory(ds_name)
                Y = [ds[i][1] for i in range(len(ds))]

                s = StratifiedShuffleSplit(
                    DS_SPLIT_CFG_NUM_SPLITS,
                    train_size=num_samples,
                    test_size=None)

                I = [i.tolist() for i, _ in s.split(Y, Y)]

                cache[(ds_name, num_samples)] = I

    with open(SPLIT_INDICES_PTH, 'bw') as fid:
        pickle.dump(cache, fid)


def ds_factory_stratified_shuffle_split(ds_name, num_samples):
    ds = ds_factory(ds_name)

    with open(SPLIT_INDICES_PTH, 'br') as fid:
        cache = pickle.load(fid)

    I = cache[ds_name, num_samples]

    return [Subset(ds, indices=i) for i in I]


verify_split_cache()
