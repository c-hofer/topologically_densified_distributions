if __name__ == '__main__':
    from pathlib import Path
    from chofer_torchex.utils.run_experiments import configs_from_grid, scatter_fn_on_devices
    from core.experiments import experiment_blue_print, experiment_jacobian, experiment_statreg
    import torch.multiprocessing as mp
    # import torch

    mp.set_start_method('spawn', force=True)

    output_root_dir = '/scratch2/chofer/cifar10_statreg_VR'

    config = {
        "output_root_dir":         [output_root_dir],
        "cv_run_num":              [10],
        "ds_train_name":           ["cifar10_train"],
        "ds_test_name":            ["cifar10_test"],
        "ds_normalization":        [True],
        "num_train_samples":       [500],
        "num_augmentations":       [1],
        "model_name":              [
             #("SimpleCNN_MNIST", {'batch_norm': True,
             #                     'cls_spectral_norm': False}),   
             ("SimpleCNN13", {'batch_norm':        True,
                              'cls_spectral_norm': False,
                              'drop_out':          True,
                              'final_bn':          True}),
        ],
        "type_augmentation":        ["standard"],
        "batch_size":              [32],
        "cls_loss_fn":             [
            "CrossEntropyLoss",
        ],
        "statreg_loss_fn":         ['VR_loss'],
        "w_statreg_loss":          [0.0001, 0.001, 0.01, 0.1],             
        "num_epochs":              [310*4],
        "lr_init":                 [0.1,0.3,0.5],
        "num_intra_samples":       [1],
        "weight_decay_feat_ext":   [0.001],
        "weight_decay_cls":        [0.0001, 0.0005, 0.001],
        "tag":                     ['cifar10_statreg_VR'],
        "track_model":             [False]
    }
    
 
    config = configs_from_grid(config)

    mp = True

    if mp:
        config = [((), c) for c in config]
        scatter_fn_on_devices(experiment_statreg, config, [2,3], 2)
