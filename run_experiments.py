if __name__ == '__main__':
    from pathlib import Path
    from chofer_torchex.utils.run_experiments import configs_from_grid, scatter_fn_on_devices
    from core.experiments import experiment_blue_print
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    output_root_dir = 'results'

   # Different configurations can be enabled here. 
   # All possible combinations of arguments in the keys of config_topo will be computed.
   # In other words a grid is executed. 
    config_topo = {
        "output_root_dir":         [output_root_dir],
        "cv_run_num":              [10],
        "ds_train_name":           [
            "cifar10_train",
            # "cifar100_train"
        ]
        "ds_test_name":            [
            "cifar10_test",
            # "cifar100_test"
        ]
        "ds_normalization":        [True],
        "num_train_samples":       [1000],
        "num_augmentations":       [1],
        "model_name":              [
             #("SimpleCNN_MNIST", {'batch_norm': True,
             #                    'cls_spectral_norm': False}),        
             ("SimpleCNN13", {'batch_norm':        True,
                              'cls_spectral_norm': False,
                              'drop_out':          True,
                              'final_bn':          True}),
        ],
        "type_augmentation":        ["standard"], # for possible values see
        # experiments.py -> AugmentingTransform 
        "batch_size":              [8],
        "cls_loss_fn":             [
            "CrossEntropyLoss",
        ],
        "num_epochs":              [310],
        "lr_init":                 [0.5],
        "num_intra_samples":       [16],
        "w_top_loss":              [0.1],
        "top_scale":               [0.7], #[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
        "weight_decay_feat_ext":   [0.001],
        "weight_decay_cls":        [0.001],
        "normalize_gradient":      [False], 
        "pers_type":               ["VrPersistenceL_2"],
        "tag":                     [''],
        "compute_persistence":     [True],
        "track_model":             [False]
    }
    
 
    config = configs_from_grid(config_topo)

    mp = False

    if mp:
        config = [((), c) for c in config]
        scatter_fn_on_devices(experiment_blue_print, config, [0], 1)

    else: 
        for c in config: 
            experiment_blue_print(**c)