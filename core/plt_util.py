import matplotlib.pyplot as plt
import torch

from chofer_torchex.utils.boiler_plate import apply_model
from .ds_util import ds_subsets_by_label


def plt_dist_to_correct_hist(dataset, model, device='cpu', ax=None):
    X, Y = apply_model(model, dataset, device=device)
    X, Y = torch.tensor(X), torch.tensor(Y)
    i_correct = (X.argmax(dim=1) == Y)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    tmp = X.softmax(dim=1)
    max = tmp.max(dim=1)[0]
    correct = torch.gather(tmp, 1, Y.unsqueeze(1)).squeeze()

    tmp_2 = max[~i_correct] - correct[~i_correct]
    tmp_2 = tmp_2.squeeze().numpy()
    ax.hist(tmp_2, alpha=0.5)


def plt_margin_hist(dataset, model, device='cpu', ax=None):
    X, Y = apply_model(model, dataset, device=device)
    X, Y = torch.tensor(X), torch.tensor(Y)
    i_correct = (X.argmax(dim=1) == Y)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    tmp = X.softmax(dim=1)
    tmp = tmp.sort(dim=1, descending=True)[0]

    for i in [i_correct, ~i_correct]:
        tmp_2 = tmp[i]
        tmp_2 = tmp_2[:, 0] - tmp_2[:, 1]
        tmp_2 = tmp_2.squeeze().numpy()
        ax.hist(tmp_2, alpha=0.5)


def plt_mapped_points_by_label(dataset, model, dim_x=0, dim_y=1, device='cpu'):

    plt.figure()

    for l, ds in ds_subsets_by_label(dataset).items():
        X, Y = apply_model(model, ds, device=device)
        X, Y = torch.tensor(X), torch.tensor(Y)
        X = X.numpy()
        plt.plot(X[:, dim_x], X[:, dim_y], '.', label=str(l))
    plt.legend()
