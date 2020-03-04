import torch
import torch.nn as nn
import chofer_torchex.nn as mynn

import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes,
            planes,
            stride=1,
            activation=nn.ReLU,
            batchnorm=nn.BatchNorm2d):

        super(BasicBlock, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = self.batchnorm(planes)
        self.act1 = self.activation()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = self.batchnorm(planes)
        self.act2 = self.activation()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                self.batchnorm(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResnetStub(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
            activation,
            batchnorm):
        super().__init__()
        self.in_planes = 64

        self.activation = activation
        self.batchnorm = batchnorm

        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                          padding=1, bias=False)
        bn1 = self.batchnorm(64)
        layer1 = self._make_layer(
            block, 64, num_blocks[0], 1, activation, batchnorm)
        layer2 = self._make_layer(
            block, 128, num_blocks[1], 2, activation, batchnorm)
        layer3 = self._make_layer(
            block, 256, num_blocks[2], 2, activation, batchnorm)
        layer4 = self._make_layer(
            block, 512, num_blocks[3], 2, activation, batchnorm)

        feat_dim = 2

        self.m = nn.Sequential(
            conv1,
            bn1,
            self.activation(),
            layer1,
            layer2,
            layer3,
            layer4,
            nn.AvgPool2d(4),
            mynn.LinearView(),
        )

    def _make_layer(
            self, block, planes, num_blocks, stride, activation, batchnorm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes,
                planes,
                stride=stride,
                activation=activation,
                batchnorm=batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.m(x)


class Resnet18(nn.Module):
    def __init__(
            self,
            num_classes,
            activation=nn.ReLU,
            batchnorm=nn.BatchNorm2d):
        super().__init__()

        self.feat_ext = ResnetStub(
            BasicBlock,
            [2, 2, 2, 2],
            activation=activation,
            batchnorm=batchnorm)

        self.cls = nn.Linear(512, num_classes)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)

        return y_hat, z


def Resnet18_no_batchnorm(num_classes):
    return Resnet18(num_classes, batchnorm=Identity)


class Resnet18IntermediateLinear(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super().__init__()

        self.feat_ext = nn.Sequential(
            ResnetStub(BasicBlock, [2, 2, 2, 2]),
            nn.Linear(512, feature_dim)
        )

        self.cls = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)

        return y_hat, z