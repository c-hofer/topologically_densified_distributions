if __name__ == '__main__':
    from chofer_torchex.utils.run_experiments import configs_from_grid, scatter_fn_on_devices
    from core.experiments import experiment_baseline
    import torch.multiprocessing as mp
    from copy import deepcopy
    
    mp.set_start_method('spawn', force=True)

    output_root_dir = '/tmp/foo'

    config = {
        "output_root_dir":         [output_root_dir],
        "cv_run_num":              [1],
        "ds_train_name":           ["cifar10_train"],
        "ds_test_name":            ["cifar10_test"],
        "ds_normalization":        [True],
        "num_train_samples":       [500],
        "model_name":              ["CNN13"],
        "typ_augmentation":        ["full_random_aug_auto"],
        "batch_size":              [0],
        "num_epochs":              [0],
        "lr_init":                 [0.1],
        "weight_decay":            [1e-3],
        "use_tb":                  [True],
        "tag":                     ["baseline"],
    }

    config = configs_from_grid(config)
    config = config[0]
    tmp = []
    for bs in [8]:
        c = deepcopy(config)
        c['batch_size'] = bs
        c['num_epochs'] = int(256/bs * 310)
        c['tag'] = "baseline_{}".format(c['num_epochs'])
        tmp.append(c)
    config = tmp

    config = [((), c) for c in config]
    scatter_fn_on_devices(experiment_baseline, config, [3], 1)
