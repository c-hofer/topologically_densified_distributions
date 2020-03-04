import torch

from chofer_torchex.pershom import vr_persistence, vr_persistence_l1

EPSILON = 0.000001


class VrPersistenceL_1:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).abs().sum(2)

        return vr_persistence(D, 0, 0)


class VrPersistenceL_2:
    def __call__(self, point_cloud):
        D = (point_cloud.unsqueeze(0) - point_cloud.unsqueeze(1)).pow(2).sum(2)
        tmp = torch.zeros_like(D)
        tmp[D == 0.0] = EPSILON
        D = D + tmp
        D = D.sqrt()
        return vr_persistence(D, 0, 0)


class VrPersistenceL_p:
    """
    IMPORTANT: This handles 0 disstance differently than VrPersistenceL_2.
    Since p < 0 is possible we have to handle zero-instances at already at 
    *coordinate* (oposed to VrPersistenceL_2 where this is done after coordinate
    summation). 
    """

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
