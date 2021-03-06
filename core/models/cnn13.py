import torch
import torch.nn as nn
import chofer_torchex.nn as mynn

import torch.nn.functional as F
from torch.nn.utils import weight_norm


def mixup_data(x, y, alpha):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class CNN13ManifoldMixup(nn.Module):
    def __init__(
            self,
            num_classes=10,
            activation=nn.LeakyReLU(0.1),
            batchnorm=nn.BatchNorm2d):
        super().__init__()

        def weight_norm(module):
            return module

        self.use_affine = True

        self.activation = activation

        # Block 1
        self.conv1a = weight_norm(self.nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = batchnorm(128, affine=self.affine)
        self.conv1a = weight_norm(self.nn.Conv2d(3, 128, 3, padding=1))
        self.bn1b = batchnorm(128, affine=self.affine)
        self.conv1c = weight_norm(self.nn.Conv2d(3, 128, 3, padding=1))
        self.bn1c = batchnorm(128, affine=self.affine)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.5)

        # Block 2
        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = batchnorm(256, affine=self.use_affine)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = batchnorm(256, affine=self.use_affine)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = batchnorm(256, affine=self.use_affine)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0),
        self.drop2 = nn.Dropout(0.5)

        # Block 3
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = batchnorm(512, affine=self.use_affine)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = batchnorm(256, affine=self.use_affine)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = batchnorm(128, affine=self.use_affine)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.lv = mynn.LinearView()

        self.cls = nn.Sequential(
            nn.Linear(128, num_classes))

    def forward(self, x, target=None, mixup_alpha=0.1, layers_mix=None, debug=False):
        layer_mix = np.random.randint(0, layers_mix)

        out = x
        out = self.conv1a(out)
        out = self.bn1a(out)
        out = self.activation(out)
        out = self.conv1b(out)
        out = self.bn1b(out)
        out = self.activation(out)
        out = self.conv1c(out)
        out = self.bn1c(out)
        if layer_mix == 1:
            out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
        out = self.activation(out)
        out = self.mp1(out)
        out = self.drop1(out)

        out = self.conv2a(out)
        out = self.bn2a(out)
        out = self.activation(out)
        out = self.conv2b(out)
        out = self.bn2b(out)
        out = self.activation(out)
        out = self.conv2c(out)
        out = self.bn2c(out)
        if layer_mix == 2:
            out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
        out = self.activation(out)
        out = self.mp2(out)
        out = self.drop2(out)

        out = self.conv3a(out)
        out = self.bn3a(out)
        out = self.activation(out)
        out = self.conv3b(out)
        out = self.bn3b(out)
        out = self.activation(out)
        out = self.conv3c(out)
        out = self.bn3c(out)
        if layer_mix == 3:
            out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
        out = self.activation(out)
        out = self.ap3(out)
        out = self.lv(out)

        lam = torch.tensor(lam).to(x.device)
        lam = lam.repeat(y_a.size())
        return self.cls(out), y_a, y_b, lam, out


class CNN13Augment(nn.Module):
    def __init__(
            self,
            num_classes=10,
            activation=nn.LeakyReLU(0.1),
            batchnorm=nn.BatchNorm2d,
            num_augmentations=None):
        super().__init__()

        def weight_norm(module):
            return module

        self.num_augmentations = num_augmentations
        assert self.num_augmentations is not None

        self.use_affine = True

        self.activation = activation

        self.beta = torch.distributions.Beta(
            torch.tensor([2.]), torch.tensor([5.]))

        # self.augmenter = nn.GRU(128, 128, 2, batch_first=True)
        # self.augmenter = nn.ModuleList([
        #                     nn.Sequential(
        #                         nn.Linear(128,10,bias=False),
        #                         nn.LeakyReLU(0.1),
        #                         nn.Linear(10,128,bias=False)) for i in range(self.num_augmentations-1)])

        # self.logvar_map = nn.Linear(128, 128)

        self.feat_ext = nn.Sequential(
            weight_norm(nn.Conv2d(3, 128, 3, padding=1)),
            batchnorm(128, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
            batchnorm(128, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
            batchnorm(128, affine=self.use_affine),
            activation,
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
            #
            weight_norm(nn.Conv2d(128, 256, 3, padding=1)),
            batchnorm(256, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
            batchnorm(256, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
            batchnorm(256, affine=self.use_affine),
            activation,
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
            #
            weight_norm(nn.Conv2d(256, 512, 3, padding=0)),
            batchnorm(512, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(512, 256, 1, padding=0)),
            batchnorm(256, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(256, 128, 1, padding=0)),
            batchnorm(128, affine=self.use_affine),
            activation,
            nn.AvgPool2d(6, stride=2, padding=0),
            mynn.LinearView()
        )

        self.cls = nn.Sequential(
            nn.Linear(128, num_classes))

    def forward(self, x, debug=False):
        z = self.feat_ext(x)
        if self.training:
            bs = z.size(0)
            z = z.unsqueeze(0).transpose(0, 1).repeat(
                1, self.num_augmentations, 1).view(-1, 128)
            index = torch.randperm(z.size(0))
            alpha = self.beta.sample(
                (self.num_augmentations*bs, 1)).view(-1, 1).to(z.device)
            noise = torch.autograd.Variable(z.data.new(
                z.size()).normal_(0, 0.01)).to(z.device)
            z = (1.0-alpha)*z + alpha*z[index, :]+noise

        # std = torch.exp(0.5*self.logvar_map(z))

        # if self.training:
        #     tmp = []
        #     for z_i, s in zip(z, std):
        #         eps = torch.randn(self.num_augmentations, 128).to(z.device)
        #         tmp.append(z_i + eps*s)
        #
        #     z = torch.stack(tmp, dim=0)
        #     z = z.reshape(z.size(0)*z.size(1), 128)

        # if self.training:
        #     tmp = []
        #     for z_i in z:
        #         tmp.append(z_i)
        #         for a in self.augmenter:
        #             tmp.append(z_i + 0.1*a(z_i))
        #
        #     z = torch.stack(tmp, dim=0)
            # z = z.unsqueeze(1).expand(-1, self.num_augmentations, -1)
            # z_r, _ = self.augmenter(z)
            # z = z + 0.1 * z_r
            # z = z.reshape(z.size(0)*z.size(1), 128)

        return self.cls(z), z


class CNN13TestStuff(nn.Module):
    """
    Alternative CNN13 model.
    """

    def __init__(
            self,
            num_classes=10,
            activation=nn.LeakyReLU(0.1),
            batchnorm=nn.BatchNorm2d):
        super().__init__()

        def weight_norm(module):
            return module

        self.activation = nn.SELU

        self.feat_ext = nn.Sequential(
            weight_norm(nn.Conv2d(3, 128, 3, padding=1)),
            # batchnorm(128),
            activation,
            weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
            # batchnorm(128),
            activation,
            weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
            # batchnorm(128),
            activation,
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
            #
            weight_norm(nn.Conv2d(128, 256, 3, padding=1)),
            # batchnorm(256),
            activation,
            weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
            # batchnorm(256),
            activation,
            weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
            # batchnorm(256),
            activation,
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
            #
            weight_norm(nn.Conv2d(256, 512, 3, padding=0)),
            # batchnorm(512),
            activation,
            weight_norm(nn.Conv2d(512, 256, 1, padding=0)),
            # batchnorm(256),
            activation,
            weight_norm(nn.Conv2d(256, 128, 1, padding=0)),
            # batchnorm(128),
            activation,
            nn.AvgPool2d(6, stride=2, padding=0),
            mynn.LinearView()
        )

        self.cls = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.BatchNorm1d(num_classes))

    def forward(self, x, debug=False):
        z = self.feat_ext(x)
        return self.cls(z), z


def cnn13_feat_ext(activation, batchnorm, bn_affine, latent_dim=None):
    return nn.Sequential(
        nn.Conv2d(3, 128, 3, padding=1),
        batchnorm(128, affine=bn_affine),
        activation,
        nn.Conv2d(128, 128, 3, padding=1),
        batchnorm(128, affine=bn_affine),
        activation,
        nn.Conv2d(128, 128, 3, padding=1),
        batchnorm(128, affine=bn_affine),
        activation,
        nn.MaxPool2d(2, stride=2, padding=0),
        nn.Dropout(0.5),
        #
        nn.Conv2d(128, 256, 3, padding=1),
        batchnorm(256, affine=bn_affine),
        activation,
        nn.Conv2d(256, 256, 3, padding=1),
        batchnorm(256, affine=bn_affine),
        activation,
        nn.Conv2d(256, 256, 3, padding=1),
        batchnorm(256, affine=bn_affine),
        activation,
        nn.MaxPool2d(2, stride=2, padding=0),
        nn.Dropout(0.5),
        #
        nn.Conv2d(256, 512, 3, padding=0),
        batchnorm(512, affine=bn_affine),
        activation,
        nn.Conv2d(512, 256, 1, padding=0),
        batchnorm(256, affine=bn_affine),
        activation,
        nn.Conv2d(256, 128, 1, padding=0),
        batchnorm(128, affine=bn_affine),
        activation,
        nn.AvgPool2d(6, stride=2, padding=0),
        mynn.LinearView(),
        Identity() if latent_dim is None else 
            nn.Sequential(
                nn.Linear(128, latent_dim),                 
                nn.BatchNorm1d(latent_dim),
                # activation 
                )
    )


class CNN13LowLatent(nn.Module):

    def __init__(
            self,
            num_classes=10,
            activation=nn.LeakyReLU(0.1),
            batchnorm=nn.BatchNorm2d):
        super().__init__()

        self.use_affine = True

        self.activation = activation

        tmp = cnn13_feat_ext(
            activation=activation,
            batchnorm=batchnorm,
            bn_affine=self.use_affine)

        self.feat_ext = nn.Sequential(
            tmp, nn.Linear(128, 16), nn.BatchNorm1d(16))

        self.cls = nn.Sequential(
            nn.Linear(16, num_classes, bias=True))

    def forward(self, x, debug=False):
        z = self.feat_ext(x)
        return self.cls(z), z


class CNN13(nn.Module):
    """
    Original CNN13 model from Mean Teacher paper (without weight_norm)
    """

    def __init__(
            self,
            num_classes=10,
            activation=nn.LeakyReLU(0.1),
            batchnorm=nn.BatchNorm2d,
            latent_dim=None):
        super().__init__()

        self.use_affine = True
        self.latent_dim = latent_dim

        self.activation = activation

        self.feat_ext = cnn13_feat_ext(
            activation=activation,
            batchnorm=batchnorm,
            bn_affine=self.use_affine,
            latent_dim=latent_dim)

        self.cls = nn.Sequential(
            nn.Linear(128 if latent_dim is None else latent_dim,
                      num_classes, bias=True)
        )

    def forward(self, x, debug=False):
        z = self.feat_ext(x)
        return self.cls(z), z


class CNN13MultipleLatent(nn.Module):
    """
    """

    def __init__(
            self,
            num_classes=10,
            activation=nn.LeakyReLU(0.1),
            batchnorm=nn.BatchNorm2d):
        super().__init__()

        self.use_affine = True

        self.activation = activation

        self.feat_ext = cnn13_feat_ext(
            activation=activation,
            batchnorm=batchnorm,
            bn_affine=self.use_affine)

        mlps = [
            nn.Sequential(
                nn.Linear(128, 32),
                nn.BatchNorm1d(32, self.use_affine),
                nn.LeakyReLU(),
                nn.Linear(32, 32)
            )
        ]

        self.mlps = nn.ModuleList(mlps)
        self.cls = nn.Linear(32, num_classes)

    @property
    def cls(self):
        return self.fc_2

    def forward(self, x, debug=False):
        latents = [self.feat_ext(x)]
        for l in self.mlps:
            latents.append(l(latents[-1]))

        return self.cls(latents[-1]), latents


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class CNN13WithBalls(nn.Module):
    """
    Original CNN13 model from Mean Teacher paper (without weight_norm) and
    alternative ball based decision rule.
    Loss is integrated in the model as otherwise the loss function would
    have to know where the centers of the balls are.
    For the same reason also the decision making is integrated in the model.
    As a consequence, the model returns in

        train mode: sample loss
        eval mode : sample prediction!

    There are always num_classes + 1 classes. The last one stands for no decision.
    """

    def __init__(
            self,
            num_classes=10,
            activation=nn.LeakyReLU(0.1),
            batchnorm=nn.BatchNorm2d):
        super().__init__()

        self.use_affine = True

        self.activation = activation
        self.latent_dim = num_classes

        feat_ext = cnn13_feat_ext(
            activation=activation,
            batchnorm=batchnorm,
            bn_affine=self.use_affine)

        feat_ext.add_module('fc', nn.Linear(128, self.latent_dim))

        self.feat_ext = feat_ext

        self.norm_p = 2
        self.ball_radius = 1

        self.register_buffer('centers', torch.eye(
            num_classes)*3.0)  # (num_classes, latent_dim)

    def forward(self, x, y=None):

        z = self.feat_ext(x)  # (bs, latent_dim)
        y_hat = self.forward_latent(z, y)

        return y_hat, z

    def forward_latent(self, z, y):
        if self.training:
            assert y is not None
            c = self.centers.index_select(0, y)  # (bs, latent_dim)
            d = z - c
            # l = d.pow(2).sum(1)
            l = d.norm(self.norm_p, dim=1)  # (bs,), --> sample loss

            return l

        else:
            assert y is None
            c = self.centers.unsqueeze(0)  # (1, num_classes, latent_dim)
            d = z.unsqueeze(1) - c
            d = d.norm(self.norm_p, dim=2)  # (bs, num_classes)
            d = (d <= self.ball_radius).long()
            u = 1 - d.sum(1).unsqueeze(1)
            d = torch.cat([d, u], dim=1)
            d = d.max(1)[1]
            return d  # (bs,)


def gaussian(ins, mean, stddev):
    noise = torch.autograd.Variable(
        ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise


class CNN13Stochastic(nn.Module):
    def __init__(
            self,
            num_classes=10,
            activation=nn.LeakyReLU(0.1),
            batchnorm=nn.BatchNorm2d):
        super().__init__()

        def weight_norm(module):
            return module

        self.use_affine = True

        self.activation = activation

        self.feat_ext = nn.Sequential(
            weight_norm(nn.Conv2d(3, 128, 3, padding=1)),
            batchnorm(128, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
            batchnorm(128, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(128, 128, 3, padding=1)),
            batchnorm(128, affine=self.use_affine),
            activation,
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
            #
            weight_norm(nn.Conv2d(128, 256, 3, padding=1)),
            batchnorm(256, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
            batchnorm(256, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(256, 256, 3, padding=1)),
            batchnorm(256, affine=self.use_affine),
            activation,
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.5),
            #
            weight_norm(nn.Conv2d(256, 512, 3, padding=0)),
            batchnorm(512, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(512, 256, 1, padding=0)),
            batchnorm(256, affine=self.use_affine),
            activation,
            weight_norm(nn.Conv2d(256, 128, 1, padding=0)),
            batchnorm(128, affine=self.use_affine),
            activation,
            nn.AvgPool2d(6, stride=2, padding=0),
            mynn.LinearView()
        )

        self.cls = nn.Sequential(
            nn.Linear(128, num_classes))

    def forward(self, x, debug=False):
        z = self.feat_ext(gaussian(x, 0, 0.15))
        return self.cls(z), z
