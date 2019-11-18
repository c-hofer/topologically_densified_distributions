import torch

from chofer_torchex.pershom import vr_persistence, vr_persistence_l1

EPSILON = 0.000001


# def vr_persistence_inf(point_cloud, *args, **kwargs):
#     D = (point_cloud.unsqueeze(0) -
#          point_cloud.unsqueeze(1)).abs().max(dim=2)[0]
#     return vr_persistence(D, 0, 0)


# def vr_persistence_l2(point_cloud, *args, **kwargs):
#     D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).pow(2).sum(2)
#     tmp = torch.zeros_like(D)
#     tmp[D == 0.0] = 0.000001
#     D = D + tmp
#     D = D.sqrt()
#     return vr_persistence(D, 0, 0)


class VrPersistenceL_p:
    def __init__(self, p):
        self.p = float(p)

    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs()

        tmp = torch.zeros_like(D)
        tmp[D < EPSILON] = EPSILON
        D = D + tmp

        D = D.pow(self.p)
        D = D.sum(2)

        D = D.pow(1./self.p)

        return vr_persistence(D, 0, 0)


class VrPersistenceL_inf:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs()
        D = D.max(dim=-1)[0]

        return vr_persistence(D, 0, 0)


class VrPersistenceF_p:
    def __init__(self, p):
        self.p = float(p)

    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs()

        tmp = torch.zeros_like(D)
        tmp[D < EPSILON] = EPSILON
        D = D + tmp

        D = D.pow(self.p)
        D = D.sum(2)

        return vr_persistence(D, 0, 0)


class VrPersistenceL_Inf:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1))
        D = D.abs().max(dim=2)[0]
        return vr_persistence(D, 0, 0)


class VrPersistenceF_0:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs()

        D = D / (1. + D)
        D = D.sum(2)

        return vr_persistence(D, 0, 0)
