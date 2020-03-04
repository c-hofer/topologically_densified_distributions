import torch
import torch.nn as nn
import chofer_torchex.nn as mynn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class SimpleCNN_MNIST(nn.Module):
    def __init__(self,
                 num_classes,
                 batch_norm,
                 cls_spectral_norm):
        super().__init__()

        def activation(): return nn.LeakyReLU(0.1)

        if batch_norm:
            def bn_2d(dim): return nn.BatchNorm2d(dim)

            def bn_1d(dim): return nn.BatchNorm1d(dim)
        else:
            def bn_2d(dim): return Identity(dim)

            def bn_1d(dim): return Identity(dim)

        self.feat_ext = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            bn_2d(8),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            #
            nn.Conv2d(8, 32, 3, padding=1),
            bn_2d(32),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            #
            nn.Conv2d(32, 64, 3, padding=1),
            bn_2d(64),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            #
            nn.Conv2d(64, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            mynn.LinearView(),
        )

        cls = nn.Linear(128, num_classes)
        if cls_spectral_norm:
            nn.utils.spectral_norm(cls)

        self.cls = nn.Sequential(cls)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)

        return y_hat, z


class SimpleCNN(nn.Module):
    def __init__(self,
                 num_classes,
                 batch_norm,
                 cls_spectral_norm):
        super().__init__()

        def activation(): return nn.LeakyReLU()

        if batch_norm:
            def bn_2d(dim): return nn.BatchNorm2d(dim)

            def bn_1d(dim): return nn.BatchNorm1d(dim)
        else:
            def bn_2d(dim): return Identity(dim)

            def bn_1d(dim): return Identity(dim)

        self.feat_ext = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            #
            nn.Conv2d(128, 256, 3, padding=1),
            bn_2d(256),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            #
            nn.Conv2d(256, 512, 3, padding=1),
            bn_2d(512),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            #
            nn.Conv2d(512, 256, 3, padding=1),
            bn_2d(256),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0), 
            #
            nn.Conv2d(256, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),    
            #
            mynn.LinearView()
        )

        cls = nn.Linear(128, num_classes)
        if cls_spectral_norm:
            nn.utils.spectral_norm(cls)

        self.cls = nn.Sequential(cls)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)

        return y_hat, z


class SimpleCNN13(nn.Module):
    def __init__(self,
                 num_classes: int,
                 batch_norm: bool, 
                 drop_out: bool, 
                 cls_spectral_norm: bool,
                 final_bn: bool):
        super().__init__()

        def activation(): return nn.LeakyReLU(0.1)

        if drop_out:
            def dropout(p): return nn.Dropout(p)
        else:
            def dropout(p): return Identity()

        bn_affine = True

        if batch_norm:
            def bn_2d(dim): return nn.BatchNorm2d(dim, affine=bn_affine)

            def bn_1d(dim): return nn.BatchNorm1d(dim, affine=bn_affine)
        else:
            def bn_2d(dim): return Identity(dim)

            def bn_1d(dim): return Identity(dim)

        self.feat_ext = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.Conv2d(128, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.Conv2d(128, 128, 3, padding=1),
            bn_2d(128),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            dropout(0.5),
            #
            nn.Conv2d(128, 256, 3, padding=1),
            bn_2d(256),
            activation(),
            nn.Conv2d(256, 256, 3, padding=1),
            bn_2d(256),
            activation(),
            nn.Conv2d(256, 256, 3, padding=1),
            bn_2d(256),
            activation(),
            nn.MaxPool2d(2, stride=2, padding=0),
            dropout(0.5),
            #
            nn.Conv2d(256, 512, 3, padding=0),
            bn_2d(512),
            activation(),
            nn.Conv2d(512, 256, 1, padding=0),
            bn_2d(256),
            activation(),
            nn.Conv2d(256, 128, 1, padding=0),
            bn_2d(128) if final_bn else Identity(),
            activation(),
            nn.AvgPool2d(6, stride=2, padding=0),
            mynn.LinearView(),
        )

        cls = nn.Linear(128, num_classes)
        if cls_spectral_norm:
            nn.utils.spectral_norm(cls)

        self.cls = nn.Sequential(cls)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)

        return y_hat, z
