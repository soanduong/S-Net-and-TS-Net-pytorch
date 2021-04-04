# -*- coding: utf-8 -*-
"""
Created on 23/06/2020 7:58 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class SNetLoss(nn.Module):
    def __init__(self, lbd=0.01):
        '''
        :param lbd: the lambda for the loss function
        '''
        super(SNetLoss, self).__init__()
        self.lbd = lbd

    def forward(self, y_fwd, y_inv, dsp_fields):
        loss = lncc_loss(y_fwd, y_inv) + self.lbd * gradient_loss(dsp_fields)
        return loss


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def lncc_loss(I, J, win=None, eps=1e-5):
    """
    This function compute the local (over window) normalized cross correlation (Baig2011) of two tensors I and J
    :param I: a tensor of size (batch, 1, 1st_size, ..., nth_size)
    :param J: a tensor of size (batch, 1, 1st_size, ..., nth_size)
    :param win:
    :return: 1- LNCC, where 0: complete correlation; and 1: no correlation.
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [2, 3], "volumes should be 2 or 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).to(I.device)

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = 1
        padding = pad_no
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding,
                                             win)

    cc = cross * cross / (I_var * J_var + eps)

    return 1 - torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross
