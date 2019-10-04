import torch

from chofer_torchex.pershom import vr_persistence, vr_persistence_l1


def vr_persistence_inf(point_cloud, *args, **kwargs):
    D = (point_cloud.unsqueeze(0) -
         point_cloud.unsqueeze(1)).abs().max(dim=2)[0]
    return vr_persistence(D, 0, 0)


def vr_persistence_l2(point_cloud, *args, **kwargs):
    D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).pow(2).sum(2)
    tmp = torch.zeros_like(D)
    tmp[D == 0.0] = 0.000001
    D = D + tmp
    D = D.sqrt()
    return vr_persistence(D, 0, 0)
