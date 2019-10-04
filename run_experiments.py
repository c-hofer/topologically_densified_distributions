if __name__ == '__main__':
    from chofer_torchex.utils.run_experiments import configs_from_grid, scatter_fn_on_devices
    from core.experiments import experiment_blue_print
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    output_root_dir = '/scratch1/chofer/topreg_2'

    config = {
        "output_root_dir":         [output_root_dir],
        "cv_run_num":              [1],
        "ds_train_name":           ["cifar10_train"],
        "ds_test_name":            ["cifar10_test"],
        "ds_normalization":        [True],
        "num_train_samples":       [250],
        "num_augmentations":       [1],
        "model_name":              ["CNN13"],
        "typ_augmentation":        ["no_aug", "full_random_aug_auto"],
        "batch_size":              [4, 8],
        "num_epochs":              [310],
        "lr_init":                 [0.1],
        "num_intra_samples":       [32],
        "w_top_loss":              [0],
        "w_top_loss_rampup_start": [-1],
        "w_top_loss_rampup_end":   [-1],
        "top_scale":               [0.5],
        "weight_decay":            [1e-3],
        "pers_type":               ["l2"],
        "use_tb":                  [True],
        "tag":                     ["explorative_no_top_loss"],
    }
    config = configs_from_grid(config)
    print('Starting {} experiments...'.format(len(config)))

    # "scheduler_kwargs":        [{'T_0'        : 10,
    #                              'T_mult'     : 2,
    #                              'eta_min'    : 0,
    #                              'last_epoch' :-1}],

#     tmp = []
#     for d in [{'batch_size': 4, 'num_intra_samples': 1}, {'batch_size': 8, 'num_intra_samples': 1}]:
#         for g in config:
#             g = dict(g)
#             g.update(d)
#             tmp.append(g)

#    config = tmp
    config = [((), c) for c in config]

    scatter_fn_on_devices(experiment_blue_print, config, [0, 1], 2)
