import uuid
import pickle
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader
from fastprogress import master_bar, progress_bar

import chofer_torchex
from chofer_torchex.utils.boiler_plate import apply_model, argmax_and_accuracy
from chofer_torchex.utils.logging import \
    convert_value_to_built_in_type as convert


import core.models as models
import core.pershom as pershom
from .ds_util import *
from .transforms import *
from .train_engine import *
from .data import ds_factory, ds_factory_stratified_shuffle_split

from .autoaugment import AutoAugment, RandomAutoAugment, FullyRandomAutoAugment, Cutout
from .augment import Augment

DEVICE = 'cuda'


def collate_fn(it):
    batch_x = []
    batch_y = []

    for x, y in it:
        batch_x.append(torch.stack(x, dim=0))
        batch_y.append(torch.tensor([y]*len(x), dtype=torch.long))

    return batch_x, batch_y


def aug_standard(to_tensor_fn):
    return transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        to_tensor_fn,
        GaussianNoise(scale=0.15),
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


def VrPersistence(id):
    d = {
        'l1': pershom.vr_persistence_l1,
        'l2': pershom.vr_persistence_l2,
        'inf': pershom.vr_persistence_inf
    }
    return d[id]


def model_factory(id, num_classes, *args, **kwargs):
    return getattr(models, id)(num_classes, *args, **kwargs)


def args_to_str(args):
    args = copy.deepcopy(args)
    del args['output_root_dir']
    del args['cv_run_num']
    del args['tensorboard_log_dir']

    return repr(args)


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
        lr_init=None,
        w_top_loss=None,
        w_top_loss_rampup_start=None,
        w_top_loss_rampup_end=None,
        top_scale=None,
        weight_decay=None,
        pers_type=None,
        tensorboard_log_dir=None,
        tag=''):

    args = dict(locals())
    print(args)
    assert all(((v is not None)
                for k, v in args.items() if k != 'tensorboard_log_dir'))
    use_tb = tensorboard_log_dir is not None

    exp_id = str(uuid.uuid4())
    output_dir = Path(output_root_dir) / exp_id
    output_dir.mkdir()

    output = {
        'args': args,
        'runs': [defaultdict(list) for _ in range(cv_run_num)]
    }

    def dump():
        with open(output_dir / 'result.pkl', 'wb') as fid:
            pickle.dump(output, fid)

    dump()

    track_accuracy = True

    """
    Get the splits for the training data.
    """
    DS_TRAIN_ORIGINAL_SPLITS = ds_factory_stratified_shuffle_split(
        ds_train_name, num_train_samples)
    DS_TEST_ORIGINAL = ds_factory(ds_test_name)
    assert len(DS_TRAIN_ORIGINAL_SPLITS) >= cv_run_num
    DS_TRAIN_ORIGINAL_SPLITS = DS_TRAIN_ORIGINAL_SPLITS[:cv_run_num]

    pers_fn = VrPersistence(pers_type)

    if use_tb:
        log_dir = Path(tensorboard_log_dir)
        writer = SummaryWriter(log_dir=log_dir / (tag + '__' + exp_id))
        writer.add_text('args', args_to_str(args))

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

        stats = output['runs'][run_i]

        model = model_factory(model_name, ds_stats['num_classes'])
        model = model.to(DEVICE)

        opt = torch.optim.SGD(model.parameters(),
                              weight_decay=weight_decay,
                              lr=lr_init,
                              momentum=0.9,
                              nesterov=True)

        # COMMENT: weight decay only on last layer
        # [
        #     {'params': model.feat_ext.parameters()},
        #     {'params': model.cls.parameters(), 'weight_decay': weight_decay}
        # ],
        # weight_decay=0,
        # lr=lr_init,
        # momentum=0.9,
        # nesterov=True)

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

        # handle the case when rampup args are not set.
        if w_top_loss_rampup_start >= 0 and w_top_loss_rampup_end > 0:
            assert w_top_loss_rampup_start < w_top_loss_rampup_end
            w_top_rampup = RampupWeight(
                start_epoch=w_top_loss_rampup_start,
                end_epoch=w_top_loss_rampup_end,
                slope=5)
        else:
            def w_top_rampup(epoch_i): return 1.0

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
                l_cls = nn.functional.cross_entropy(y_hat, y)

                l_top = torch.tensor(0.0).to(DEVICE)

                if False:
                    for i in range(batch_size):
                        z_sample = z[i*n:(i+1)*n, :].contiguous()
                        lt = pers_fn(z_sample, 0, 0)[0][0][:, 1]

                        if use_tb:
                            writer.add_scalar(
                                'run_{:d}/avg_lt_sample'.format(run_i),
                                convert(lt.mean()))
                            with torch.no_grad():
                                writer.add_scalar(
                                    'run_{:d}/avg_sample_norm'.format(run_i),
                                    convert(z_sample.norm(dim=1).mean()))

                        stats['batch_lt'].append(convert(lt))
                        l_top = l_top + (lt-top_scale).abs().sum()
                        # l_top = (z_sample.var(dim=0)-top_scale).abs().sum()
                    l_top = l_top / float(batch_size)

                l = l_cls + w_top_loss * l_top * w_top_rampup(epoch_i)

                if use_tb:
                    writer.add_scalar('run_{:d}/learning_rate'.format(run_i),
                                      convert(scheduler.get_lr()[0]),
                                      epoch_i + b_i / L)

                stats['lr'].append(convert(scheduler.get_lr()[0]))

                opt.zero_grad()
                l.backward()
                opt.step()

                if use_tb:

                    for k, p in model.named_parameters():
                        p_norm = convert(p.data.norm())
                        d_p_norm = convert(p.grad.data.norm())

                        writer.add_scalar(
                            'run_{:d}/norm_p_{}'.format(run_i, k),
                            p_norm)

                        writer.add_scalar(
                            'run_{:d}/norm_grad_p_{}'.format(run_i, k),
                            d_p_norm)

                        # param_norm = torch.tensor([p.norm() for p in model.parameters()])
                        # writer.add_scalar(
                        #     'run_{:d}/sum_parameter_norm'.format(run_i),
                        #     convert(param_norm.sum()))

                        # param_norm = torch.tensor([p.grad.norm() for p in model.parameters()])
                        # writer.add_scalar(
                        #     'run_{:d}/sum_parameter_grad_norm'.format(run_i),
                        #     convert(param_norm.sum()))

                epoch_loss += l.item()
                stats['batch_cls_loss'].append(convert(l_cls))
                stats['batch_top_loss'].append(convert(l_top))

                if use_tb:
                    writer.add_scalar(
                        'run_{:d}/cls_loss'.format(run_i), convert(l_cls))
                    writer.add_scalar(
                        'run_{:d}/top_loss'.format(run_i), convert(l_top))

            scheduler.step()

            mb_comment = "Last loss: {:.2f} {:.4f} ".format(
                epoch_loss,
                w_top_loss*w_top_rampup(epoch_i))

            if track_accuracy:

                X, Y = apply_model(model, DS_TRAIN, device=DEVICE)
                acc_train = argmax_and_accuracy(X, Y)
                stats['acc_train'].append(convert(acc_train))
                mb_comment += " | acc. train {:.2f} ".format(acc_train)

                X, Y = apply_model(model, DS_TEST, device=DEVICE)
                acc_test = argmax_and_accuracy(X, Y)
                stats['acc_test'].append(convert(acc_test))
                mb_comment += " | acc. test {:.2f} ".format(acc_test)

                if use_tb:
                    writer.add_scalar(
                        'run_{:d}/Train'.format(run_i), acc_train, epoch_i)
                    writer.add_scalar(
                        'run_{:d}/Test'.format(run_i), acc_test, epoch_i)

                mb.first_bar.comment = mb_comment

        dump()
        torch.save(model, output_dir / 'model_run_{}.pth'.format(run_i))

    if use_tb:
        writer.flush()
        writer.close()


def experiment_no_latent_space(
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
        lr_init=None,
        w_top_loss=None,
        w_top_loss_rampup_start=None,
        w_top_loss_rampup_end=None,
        top_scale=None,
        weight_decay=None,
        pers_type=None,
        tensorboard_log_dir=None,
        tag=''):

    args = dict(locals())
    print(args)
    assert all(((v is not None)
                for k, v in args.items() if k != 'tensorboard_log_dir'))
    use_tb = tensorboard_log_dir is not None

    exp_id = str(uuid.uuid4())
    output_dir = Path(output_root_dir) / exp_id
    output_dir.mkdir()

    output = {
        'args': args,
        'runs': [defaultdict(list) for _ in range(cv_run_num)]
    }

    def dump():
        with open(output_dir / 'result.pkl', 'wb') as fid:
            pickle.dump(output, fid)

    dump()

    track_accuracy = True

    """
    Get the splits for the training data.
    """
    DS_TRAIN_ORIGINAL_SPLITS = ds_factory_stratified_shuffle_split(
        ds_train_name, num_train_samples)
    DS_TEST_ORIGINAL = ds_factory(ds_test_name)
    assert len(DS_TRAIN_ORIGINAL_SPLITS) >= cv_run_num
    DS_TRAIN_ORIGINAL_SPLITS = DS_TRAIN_ORIGINAL_SPLITS[:cv_run_num]

    pers_fn = VrPersistence(pers_type)

    if use_tb:
        log_dir = Path(tensorboard_log_dir)
        writer = SummaryWriter(log_dir=log_dir / (tag + '__' + exp_id))
        writer.add_text('args', args_to_str(args))

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

        stats = output['runs'][run_i]

        model = model_factory(model_name, ds_stats['num_classes'])
        model = model.to(DEVICE)

        opt = torch.optim.SGD(model.parameters(),
                              weight_decay=weight_decay,
                              lr=lr_init,
                              momentum=0.9,
                              nesterov=True)

        # COMMENT: weight decay only on last layer
        # [
        #     {'params': model.feat_ext.parameters()},
        #     {'params': model.cls.parameters(), 'weight_decay': weight_decay}
        # ],
        # weight_decay=0,
        # lr=lr_init,
        # momentum=0.9,
        # nesterov=True)

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

        # handle the case when rampup args are not set.
        if w_top_loss_rampup_start >= 0 and w_top_loss_rampup_end > 0:
            assert w_top_loss_rampup_start < w_top_loss_rampup_end
            w_top_rampup = RampupWeight(
                start_epoch=w_top_loss_rampup_start,
                end_epoch=w_top_loss_rampup_end,
                slope=5)
        else:
            def w_top_rampup(epoch_i): return 1.0

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
                l_cls = nn.functional.cross_entropy(y_hat, y)

                l_top = torch.tensor(0.0).to(DEVICE)

                if True:
                    for i in range(batch_size):
                        y_hat_sample = y_hat[i*n:(i+1)*n, :].contiguous()
                        lt = pers_fn(y_hat, 0, 0)[0][0][:, 1]

                        if use_tb:
                            writer.add_scalar(
                                'run_{:d}/avg_lt_sample'.format(run_i),
                                convert(lt.mean()))
                            with torch.no_grad():
                                writer.add_scalar(
                                    'run_{:d}/avg_sample_norm'.format(run_i),
                                    convert(y_hat_sample.norm(dim=1).mean()))

                        stats['batch_lt'].append(convert(lt))
                        l_top = l_top + (lt-top_scale).abs().sum()
                        # l_top = (z_sample.var(dim=0)-top_scale).abs().sum()
                    l_top = l_top / float(batch_size)

                l = l_cls + w_top_loss * l_top * w_top_rampup(epoch_i)

                if use_tb:
                    writer.add_scalar('run_{:d}/learning_rate'.format(run_i),
                                      convert(scheduler.get_lr()[0]),
                                      epoch_i + b_i / L)

                stats['lr'].append(convert(scheduler.get_lr()[0]))

                opt.zero_grad()
                l.backward()
                opt.step()

                if use_tb:

                    for k, p in model.named_parameters():
                        p_norm = convert(p.data.norm())
                        d_p_norm = convert(p.grad.data.norm())

                        writer.add_scalar(
                            'run_{:d}/norm_p_{}'.format(run_i, k),
                            p_norm)

                        writer.add_scalar(
                            'run_{:d}/norm_grad_p_{}'.format(run_i, k),
                            d_p_norm)

                        # param_norm = torch.tensor([p.norm() for p in model.parameters()])
                        # writer.add_scalar(
                        #     'run_{:d}/sum_parameter_norm'.format(run_i),
                        #     convert(param_norm.sum()))

                        # param_norm = torch.tensor([p.grad.norm() for p in model.parameters()])
                        # writer.add_scalar(
                        #     'run_{:d}/sum_parameter_grad_norm'.format(run_i),
                        #     convert(param_norm.sum()))

                epoch_loss += l.item()
                stats['batch_cls_loss'].append(convert(l_cls))
                stats['batch_top_loss'].append(convert(l_top))

                if use_tb:
                    writer.add_scalar(
                        'run_{:d}/cls_loss'.format(run_i), convert(l_cls))
                    writer.add_scalar(
                        'run_{:d}/top_loss'.format(run_i), convert(l_top))

            scheduler.step()

            mb_comment = "Last loss: {:.2f} {:.4f} ".format(
                epoch_loss,
                w_top_loss*w_top_rampup(epoch_i))

            if track_accuracy:

                X, Y = apply_model(model, DS_TRAIN, device=DEVICE)
                acc_train = argmax_and_accuracy(X, Y)
                stats['acc_train'].append(convert(acc_train))
                mb_comment += " | acc. train {:.2f} ".format(acc_train)

                X, Y = apply_model(model, DS_TEST, device=DEVICE)
                acc_test = argmax_and_accuracy(X, Y)
                stats['acc_test'].append(convert(acc_test))
                mb_comment += " | acc. test {:.2f} ".format(acc_test)

                if use_tb:
                    writer.add_scalar(
                        'run_{:d}/Train'.format(run_i), acc_train, epoch_i)
                    writer.add_scalar(
                        'run_{:d}/Test'.format(run_i), acc_test, epoch_i)

                mb.first_bar.comment = mb_comment

        dump()
        torch.save(model, output_dir / 'model_run_{}.pth'.format(run_i))

    if use_tb:
        writer.flush()
        writer.close()


def experiment_multiple_latent(
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
        lr_init=None,
        w_top_loss=None,
        w_top_loss_rampup_start=None,
        w_top_loss_rampup_end=None,
        top_scale=None,
        weight_decay=None,
        pers_type=None,
        tensorboard_log_dir=None,
        tag=''):

    args = dict(locals())
    print(args)
    assert all(((v is not None)
                for k, v in args.items() if k != 'tensorboard_log_dir'))
    use_tb = tensorboard_log_dir is not None

    exp_id = str(uuid.uuid4())
    output_dir = Path(output_root_dir) / exp_id
    output_dir.mkdir()

    output = {
        'args': args,
        'runs': [defaultdict(list) for _ in range(cv_run_num)]
    }

    def dump():
        with open(output_dir / 'result.pkl', 'wb') as fid:
            pickle.dump(output, fid)

    dump()

    track_accuracy = True

    """
    Get the splits for the training data.
    """
    DS_TRAIN_ORIGINAL_SPLITS = ds_factory_stratified_shuffle_split(
        ds_train_name, num_train_samples)
    DS_TEST_ORIGINAL = ds_factory(ds_test_name)
    assert len(DS_TRAIN_ORIGINAL_SPLITS) >= cv_run_num
    DS_TRAIN_ORIGINAL_SPLITS = DS_TRAIN_ORIGINAL_SPLITS[:cv_run_num]

    pers_fn = VrPersistence(pers_type)

    if use_tb:
        log_dir = Path(tensorboard_log_dir)
        writer = SummaryWriter(log_dir=log_dir / (tag + '__' + exp_id))
        writer.add_text('args', args_to_str(args))

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

        stats = output['runs'][run_i]

        model = model_factory(model_name, ds_stats['num_classes'])
        model = model.to(DEVICE)

        opt = torch.optim.SGD(model.parameters(),
                              weight_decay=weight_decay,
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

        # handle the case when rampup args are not set.
        if w_top_loss_rampup_start >= 0 and w_top_loss_rampup_end > 0:
            assert w_top_loss_rampup_start < w_top_loss_rampup_end
            w_top_rampup = RampupWeight(
                start_epoch=w_top_loss_rampup_start,
                end_epoch=w_top_loss_rampup_end,
                slope=5)
        else:
            def w_top_rampup(epoch_i): return 1.0

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

                y_hat, latents = model(x)
                l_cls = nn.functional.cross_entropy(y_hat, y)

                if True:
                    assert isinstance(latents, list)

                    z_1, z_2 = latents

                    # handling 1st latent space
                    l_top_1 = torch.tensor(0.0).to(DEVICE)
                    for i in range(batch_size):
                        z_sample = z_2[i*n:(i+1)*n, :].contiguous()
                        lt = pers_fn(z_sample, 0, 0)[0][0][:, 1]

                        stats['batch_lt_latent_1'].append(convert(lt))
                        l_top_1 = l_top_1 + (lt-top_scale).abs().sum()

                    l_top_1 = l_top_1 / float(batch_size)

                    # handling 2nd latent space
                    l_top_2 = torch.tensor(0.0).to(DEVICE)
                    for i in range(batch_size):
                        z_sample = z_2[i*n:(i+1)*n, :].contiguous()
                        lt = pers_fn(z_sample, 0, 0)[0][0][:, 1]

                        stats['batch_lt_latent_2'].append(convert(lt))
                        
                        l_top_2 = l_top_2 +lt.sort()[0][:num_intra_samples*num_augmentations//2].sum()

                    l_top_2 = l_top_2 / float(batch_size)

                l = l_cls + w_top_loss * (l_top_1 + l_top_2)

                if use_tb:
                    writer.add_scalar('run_{:d}/learning_rate'.format(run_i),
                                      convert(scheduler.get_lr()[0]),
                                      epoch_i + b_i / L)

                stats['lr'].append(convert(scheduler.get_lr()[0]))

                opt.zero_grad()
                l.backward()
                opt.step()

                # if use_tb:

                # for k, p in model.named_parameters():
                #     p_norm = convert(p.data.norm())
                #     d_p_norm = convert(p.grad.data.norm())

                #     writer.add_scalar(
                #         'run_{:d}/norm_p_{}'.format(run_i, k),
                #         p_norm)

                #     writer.add_scalar(
                #         'run_{:d}/norm_grad_p_{}'.format(run_i, k),
                #         d_p_norm)

                epoch_loss += l.item()
                stats['batch_cls_loss'].append(convert(l_cls))
                stats['batch_top_loss_1'].append(convert(l_top_1))
                stats['batch_top_loss_2'].append(convert(l_top_2))

                if use_tb:
                    writer.add_scalar(
                        'run_{:d}/cls_loss'.format(run_i), convert(l_cls))
                    writer.add_scalar(
                        'run_{:d}/top_loss_1'.format(run_i), convert(l_top_1))
                    writer.add_scalar(
                        'run_{:d}/top_loss_2'.format(run_i), convert(l_top_2))

            scheduler.step()

            mb_comment = "Last loss: {:.2f} {:.4f} ".format(
                epoch_loss,
                w_top_loss*w_top_rampup(epoch_i))

            if track_accuracy:

                X, Y = apply_model(model, DS_TRAIN, device=DEVICE)
                acc_train = argmax_and_accuracy(X, Y)
                stats['acc_train'].append(convert(acc_train))
                mb_comment += " | acc. train {:.2f} ".format(acc_train)

                X, Y = apply_model(model, DS_TEST, device=DEVICE)
                acc_test = argmax_and_accuracy(X, Y)
                stats['acc_test'].append(convert(acc_test))
                mb_comment += " | acc. test {:.2f} ".format(acc_test)

                if use_tb:
                    writer.add_scalar(
                        'run_{:d}/Train'.format(run_i), acc_train, epoch_i)
                    writer.add_scalar(
                        'run_{:d}/Test'.format(run_i), acc_test, epoch_i)

                mb.first_bar.comment = mb_comment

        dump()
        torch.save(model, output_dir / 'model_run_{}.pth'.format(run_i))

    if use_tb:
        writer.flush()
        writer.close()
