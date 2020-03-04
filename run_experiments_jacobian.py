if __name__ == '__main__':
    from pathlib import Path
    from chofer_torchex.utils.run_experiments import configs_from_grid, scatter_fn_on_devices
    from core.experiments import experiment_blue_print, experiment_jacobian
    import torch.multiprocessing as mp
    # import torch

    mp.set_start_method('spawn', force=True)

    output_root_dir = '/scratch2/chofer/cifar10_vanilla_w_jacobian'

    config_topo = {
        "output_root_dir":         [output_root_dir],
        "cv_run_num":              [10],
        "ds_train_name":           ["cifar10_train"],
        "ds_test_name":            ["cifar10_test"],
        "ds_normalization":        [True],
        "num_train_samples":       [500],
        "num_augmentations":       [1],
        "model_name":              [
            ("SimpleCNN13", {'batch_norm': True,
                             'cls_spectral_norm': False,
                             'drop_out': True,
                             'final_bn': True}),
        ],
        "type_augmentation":        ["standard"],
        "batch_size":              [32],
        "cls_loss_fn":             [
            "CrossEntropyLoss",
        ],
        "num_epochs":              [310*4],
        "lr_init":                 [0.1, 0.3, 0.5],
        "num_intra_samples":       [1],
        "w_top_loss":              [0.],
        "top_scale":               [0.],
        "weight_decay_feat_ext":   [0.001],
        "weight_decay_cls":        [0.0005,0.0001],
        "lambda_JR":               [1e-3,1e-4],
        "normalize_gradient":      [False],
        "pers_type":               ["VrPersistenceL_2"],
        "tag":                     ['CIFAR10_jacobian'],
        "compute_persistence":     [False],
        "track_model":             [False]
    }

    config = configs_from_grid(config_topo)

    mp = True

    if mp:
        config = [((), c) for c in config]
        scatter_fn_on_devices(experiment_jacobian, config, [0,1], 3)
