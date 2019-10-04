import torch
import torchvision

from collections import defaultdict, OrderedDict


def ds_statistics(ds):
    x, _ = ds[0]
    if not isinstance(x, torch.Tensor):
        ds = Transformer(ds, torchvision.transforms.ToTensor())

    X = []
    Y = []
    for i in range(len(ds)):
        x, y = ds[i]
        x = x.view(x.size(0), -1)
        X.append(x)
        Y.append(y)

    X = torch.cat(X, dim=1)

    mean = tuple(X.mean(dim=1).tolist())
    std = tuple(X.std(dim=1).tolist())

    num_classes = len(set(Y))

    return {
        'channel_mean': mean, 
        'channel_std': std, 
        'num_classes': num_classes
    }


def ds_subsets_by_label(dataset):
    i_by_label = defaultdict(list)
    for i, (_, y) in enumerate(dataset):
        i_by_label[int(y)].append(i)

    ret = OrderedDict({
        l: Subset(dataset, i_by_label[l]) for l in sorted(i_by_label.keys())
    })

    return ret


class DynamicDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, wrappee):
        super().__init__()
        assert isinstance(wrappee, torch.utils.data.Dataset)
        self.wrappee = wrappee

    def __getattr__(self, name):
        return getattr(self.__dict__['wrappee'], name)

    def __len__(self):
        return len(self.wrappee)

    def __getitem__(self, idx):
        return self.wrappee[idx]


class Transformer(DynamicDatasetWrapper):
    def __init__(self, wrappee, transform=None, target_transform=None):
        super().__init__(wrappee)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x, y = self.wrappee[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


class RepeatedAugmentation(DynamicDatasetWrapper):
    def __init__(self, ds, transform, num_augmentations):
        self.transform = transform
        self.num_aug = num_augmentations
        self.wrappee = ds
        self.augmented_wrappee = Transformer(ds, transform=transform)

    def __getitem__(self, idx):
        x, y = self.augmented_wrappee[idx]
        x = [x] + [self.augmented_wrappee[idx][0]
                   for _ in range(self.num_aug-1)]
        return x, y

    def __len__(self):
        return len(self.wrappee)


class IntraLabelMultiDraw(DynamicDatasetWrapper):
    def __init__(self, ds, num_draws):
        assert isinstance(ds, torch.utils.data.Dataset)

        self.wrappee = ds
        self.num_draws = int(num_draws)
        assert self.num_draws > 0

        self.indices_by_label = defaultdict(list)
        for i in range(len(ds)):
            _, y = ds[i]
            y = int(y)
            self.indices_by_label[y].append(i)

    def __getitem__(self, idx):
        x, y = self.wrappee[idx]
        assert isinstance(x, list)

        N = len(self.indices_by_label[y])
        I = torch.randint(N, (self.num_draws - 1,))
        I = [self.indices_by_label[y][i] for i in I]

        for i in I:
            x_i, y_i = self.wrappee[i]
            x += x_i

        return x, y

    def __len__(self):
        return len(self.wrappee)
