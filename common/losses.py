########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_list():
    return loss_list.keys()


def get_loss_fn(args):
    return loss_list[args.loss]
    return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:,:1,:,:]
        return F.l1_loss(outputs, target)


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:,:1,:,:]
        return F.mse_loss(outputs, target)


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:,:1,:,:]
        return F.smooth_l1_loss(outputs, target)


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:,:1,:,:]
        val_pixels = torch.ne(target, 0).float().detach()
        return F.l1_loss(outputs*val_pixels, target*val_pixels)


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:,:1,:,:]
        val_pixels = torch.ne(target, 0).float().detach()
        return F.mse_loss(outputs*val_pixels, target*val_pixels)


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:,:1,:,:]
        val_pixels = torch.ne(target, 0).float().detach()
        loss = F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        return torch.mean(loss)


# The proposed probabilistic loss for pNCNN
class MaskedProbLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, targets):
        means = out[:, :1, :, :]
        cout = out[:, 1:2, :, :]

        res = cout
        regl = torch.log(cout+1e-16)  # Regularization term

        # Pick only valid pixels
        valid_mask = (targets > 0).detach()
        targets = targets[valid_mask]
        means = means[valid_mask]
        res = res[valid_mask]
        regl = regl[valid_mask]

        loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        return loss


class MaskedProbExpLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, targets):
        means = out[:, :1, :, :]
        cout = out[:, 1:2, :, :]

        res = torch.exp(cout)  # Residual term
        regl = torch.log(cout+1e-16)  # Regularization term

        # Pick only valid pixels
        valid_mask = (targets > 0).detach()
        targets = targets[valid_mask]
        means = means[valid_mask]
        res = res[valid_mask]
        regl = regl[valid_mask]

        loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        return loss

loss_list = {
    'l1': L1Loss(),
    'l2': L2Loss(),
    'masked_l1': MaskedL1Loss(),
    'masked_l2': MaskedL2Loss(),
    'masked_prob_loss': MaskedProbLoss(),
    'masked_prob_exp_loss': MaskedProbExpLoss(),
}
