import uuid
import pickle
import json
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
from pathlib import Path
from datetime import datetime

from torch.utils.data import DataLoader
from fastprogress import master_bar, progress_bar

import chofer_torchex
from chofer_torchex.utils.boiler_plate import apply_model, argmax_and_accuracy
from chofer_torchex.utils.logging import \
    convert_value_to_built_in_type as convert
from chofer_torchex.utils.data.ds_operations import ds_random_subset


import core.models as models
import core.pershom as pershom
from .ds_util import *
from .transforms import *
from .train_engine import *
from .data import ds_factory, ds_factory_stratified_shuffle_split

from .autoaugment import AutoAugment, RandomAutoAugment, FullyRandomAutoAugment, Cutout
from .augment import Augment

DEVICE = 'cuda'


def keychain_value_iter(d, key_chain=None, allowed_values=None):
    key_chain = [] if key_chain is None else list(key_chain).copy()

    if not isinstance(d, dict):
        if allowed_values is not None:
            assert isinstance(d, allowed_values), 'Value needs to be of type {}!'.format(
                allowed_values)
        yield key_chain, d
    else:
        for k, v in d.items():
            yield from keychain_value_iter(
                v,
                key_chain + [k],
                allowed_values=allowed_values)


def collate_fn(it):
    batch_x = []
    batch_y = []

    for x, y in it:
        batch_x.append(torch.stack(x, dim=0))
        batch_y.append(torch.tensor([y]*len(x), dtype=torch.long))

    return batch_x, batch_y


def aug_standard(to_tensor_fn):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        to_tensor_fn,
    ])


def aug_auto(to_tensor_fn):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(),
        to_tensor_fn
    ])


def random_aug_auto(to_tensor_fn):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandomAutoAugment(),
        to_tensor_fn
    ])


def full_random_aug_auto(to_tensor_fn):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        FullyRandomAutoAugment(),
        to_tensor_fn
    ])


def custom_aug(to_tensor_fn):
    return transforms.Compose([
        Augment(),
        to_tensor_fn
    ])


def no_aug(to_tensor_fn):
    return to_tensor_fn


def AugmentingTransform(id, to_tensor_fn):
    d = {
        'standard': aug_standard,
        'aug_auto': aug_auto,
        'random_aug_auto': random_aug_auto,
        'full_random_aug_auto': full_random_aug_auto,
        'custom_aug': custom_aug,
        'no_aug': no_aug
    }

    return d[id](to_tensor_fn)


def persistence_fn_factory(arg):

    if isinstance(arg, str):
        return getattr(pershom, arg)()
    elif isinstance(arg, tuple):
        id, kwargs = arg
        return getattr(pershom, id)(**kwargs)
    else:
        raise ValueError()


def model_factory(arg, num_classes):
    if isinstance(arg, str):
        return getattr(models, arg)(num_classes)
    elif isinstance(arg, tuple):
        id, kwargs = arg
        return getattr(models, id)(num_classes, **kwargs)
    else:
        raise ValueError()


def cls_loss_fn_factory(arg):
    if isinstance(arg, str):
        return getattr(nn, arg)()
    elif isinstance(arg, tuple):
        id, kwargs = arg
        return getattr(nn, id)(**kwargs)
    else:
        raise ValueError()


def get_experiment_id(tag):
    exp_id = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    if tag != '':
        exp_id = '__'.join([exp_id, tag])

    exp_id = '__'.join([exp_id,  str(uuid.uuid4())])

    return exp_id


def dump_results(dir: str, separator='__'):
    assert all((separator not in k for k in result))

    for k, v in keychain_value_iter(dict):
        path = Path(dir) / '__'.join(k)
        print(path)
        with open(str(path) + '.pkl', 'bw') as fid:
            pickle.dump(obj=v, file=fid)


class ExperimentLogger():
    def __init__(self, log_dir, experiment_args: dict):
        self.log_dir = Path(log_dir)
        self.current_cv_run = 0

        with open(self.log_dir / 'args.json', 'w') as fid:
            json.dump(obj=experiment_args, fp=fid)

        self._value_buffer = None

    def new_run(self):
        if self.current_cv_run > 0:
            self.write_logged_values_to_disk()

        self.current_cv_run += 1
        self._current_write_dir = self.log_dir / str(self.current_cv_run)
        self._current_write_dir.mkdir()
        self._value_buffer = defaultdict(list)

    def log_value(self, key: str, value):
        assert isinstance(key, str)
        assert self._value_buffer is not None
        v = convert(value)
        self._value_buffer[key].append(v)

    def write_logged_values_to_disk(self):
        for k, v in self._value_buffer.items():
            pth = self._current_write_dir / (k + '.pkl')
            with open(pth, 'bw') as fid:
                pickle.dump(obj=v, file=fid)

    def write_model_to_disk(self, key: str, model):
        assert isinstance(key, str)
        assert self.current_cv_run > 0

        torch.save(model, self._current_write_dir / (key + '.pth'))


def experiment_blue_print(
        output_root_dir=None,
        cv_run_num=None,
        ds_train_name=None,
        ds_test_name=None,
        ds_normalization=None,
        num_train_samples=None,
        num_augmentations=None,
        typ_augmentation=None,
        num_intra_samples=None,
        model_name=None,
        batch_size=None,
        num_epochs=None,
        cls_loss_fn=None,
        lr_init=None,
        w_top_loss=None,
        top_scale=None,
        weight_decay_cls=None,
        weight_decay_feat_ext=None,
        normalize_gradient=None,
        pers_type=None,
        compute_persistence=None,
        track_model=None,
        tag=''):

    args = dict(locals())
    print(args)
    if not all(((v is not None) for k, v in args.items())):
        s = ', '.join((k for k, v in args.items() if v is None))
        raise AssertionError("Some kwargs are None: {}!".format(s))

    if w_top_loss > 0 and not compute_persistence:
        raise AssertionError('w_top_loss > 0 and compute_persistence == False')

    exp_id = get_experiment_id(tag)
    output_dir = Path(output_root_dir) / exp_id
    output_dir.mkdir()

    logger = ExperimentLogger(output_dir, args)

    track_accuracy = True

    """
    Get the splits for the training data.
    """
    DS_TRAIN_ORIGINAL_SPLITS = ds_factory_stratified_shuffle_split(
        ds_train_name, num_train_samples)
    DS_TEST_ORIGINAL = ds_factory(ds_test_name)
    assert len(DS_TRAIN_ORIGINAL_SPLITS) >= cv_run_num
    DS_TRAIN_ORIGINAL_SPLITS = DS_TRAIN_ORIGINAL_SPLITS[:cv_run_num]

    pers_fn = persistence_fn_factory(args['pers_type'])
    cls_loss_fn = cls_loss_fn_factory(args['cls_loss_fn'])

    """
    Run over the dataset splits; the splits are fixed for each number of
    training samples (500,1000,4000, etc.)
    """
    for run_i, DS_TRAIN_ORIGINAL in enumerate(DS_TRAIN_ORIGINAL_SPLITS):

        t = [transforms.ToTensor()]
        ds_stats = ds_statistics(DS_TRAIN_ORIGINAL)
        if ds_normalization:
            t += [transforms.Normalize(
                ds_stats['channel_mean'],
                ds_stats['channel_std'])]
        to_tensor = transforms.Compose(t)

        augmenting_transform = AugmentingTransform(typ_augmentation, to_tensor)
        DS_TRAIN = Transformer(DS_TRAIN_ORIGINAL, transform=to_tensor)
        DS_TRAIN_AUGMENTED = RepeatedAugmentation(
            DS_TRAIN_ORIGINAL, augmenting_transform, num_augmentations)
        DS_TRAIN_AUGMENTED = IntraLabelMultiDraw(
            DS_TRAIN_AUGMENTED, num_intra_samples)
        DS_TEST = Transformer(DS_TEST_ORIGINAL, transform=to_tensor)
        assert len(DS_TRAIN_ORIGINAL) == num_train_samples

        logger.new_run()

        model = model_factory(model_name, ds_stats['num_classes'])
        model = model.to(DEVICE)
        print(model)

        opt = torch.optim.SGD(
            [
                {'params': model.feat_ext.parameters(
                ), 'weight_decay': weight_decay_feat_ext},
                {'params': model.cls.parameters(), 'weight_decay': weight_decay_cls}
            ],
            lr=lr_init,
            momentum=0.9,
            nesterov=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=num_epochs,
            eta_min=0,
            last_epoch=-1)

        dl_train = DataLoader(
            DS_TRAIN_AUGMENTED,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=0)

        mb = master_bar(range(num_epochs))
        mb_comment = ''

        for epoch_i in mb:

            model.train()
            epoch_loss = 0

            L = len(dl_train)-1
            for b_i, ((batch_x, batch_y), _) in enumerate(zip(dl_train, progress_bar(range(L), parent=mb))):

                n = batch_x[0].size(0)
                assert n == num_intra_samples*num_augmentations
                assert all(((x.size(0) == n) for x in batch_x))

                x, y = torch.cat(batch_x, dim=0), torch.cat(batch_y, dim=0)
                x, y = x.to(DEVICE), y.to(DEVICE)

                y_hat, z = model(x)
                l_cls = cls_loss_fn(y_hat, y)

                l_top = torch.tensor(0.0).to(DEVICE)

                if compute_persistence:
                    for i in range(batch_size):
                        z_sample = z[i*n: (i+1)*n, :].contiguous()
                        lt = pers_fn(z_sample)[0][0][:, 1]

                        logger.log_value('batch_lt', lt)
                        l_top = l_top + (lt-top_scale).abs().sum()
                    l_top = l_top / float(batch_size)

                 l = l_cls + w_top_loss * l_top

                opt.zero_grad()
                l.backward()

                # gradient norm and normalization aa
                grad_vec_abs = torch.cat(
                    [p.grad.data.view(-1) for p in model.parameters()], dim=0).abs()

                grad_norm = grad_vec_abs.pow(2).sum().sqrt().item()

                if grad_norm > 0 and normalize_gradient:
                    for p in model.parameters():
                        p.grad.data /= grad_norm

                opt.step()

                epoch_loss += l.item()
                logger.log_value('batch_cls_loss', l_cls)
                logger.log_value('batch_top_loss', l_top)

                logger.log_value('batch_grad_norm', grad_norm)
                logger.log_value('batch_grad_abs_max', grad_vec_abs.max())
                logger.log_value('batch_grad_abs_min', grad_vec_abs.min())
                logger.log_value('batch_grad_abs_mean', grad_vec_abs.mean())
                logger.log_value('batch_grad_abs_std', grad_vec_abs.std())

                logger.log_value('lr', scheduler.get_lr()[0])
                logger.log_value(
                    'cls_norm', model.cls[0].weight.data.view(-1).norm())

            scheduler.step()

            mb_comment = "Last loss: {:.2f} {:.4f} ".format(
                epoch_loss,
                w_top_loss)

            if track_accuracy:

                X, Y = apply_model(model, DS_TRAIN, device=DEVICE)
                acc_train = argmax_and_accuracy(X, Y)
                logger.log_value('acc_train', acc_train)
                mb_comment += " | acc. train {:.2f} ".format(acc_train)

                X, Y = apply_model(model, DS_TEST, device=DEVICE)
                acc_test = argmax_and_accuracy(X, Y)
                logger.log_value('acc_test', acc_test)
                mb_comment += " | acc. test {:.2f} ".format(acc_test)

                logger.log_value('epoch_i', epoch_i)

                mb.first_bar.comment = mb_comment

            logger.write_logged_values_to_disk()

            if track_model:
                logger.write_model_to_disk('model_epoch_{}'.format(epoch_i),
                                           model)

        logger.write_model_to_disk('model', model)
