# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************

import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math
import torch.nn as nn
from layers import SpatialTransformer, shape, ResizeTransform
import ext.pynd as pynd
from mind import MIND
from torch import Tensor


class LossFunction_mpr(nn.Module):
  def __init__(self, shape):
    super(LossFunction_mpr, self).__init__()

    self.shape = shape
    self.ncc_loss = ncc_loss()
    self.mind_loss =MIND()
    self.gradient_loss = gradient_loss()
    self.multi_loss = multi_loss(self.shape)

  def forward(self, y, tgt, src, flow, flow1, flow2,
              hyper_1=10, hyper_2=15, hyper_3=3.2, hyper_4=0.8, hyper_5=0.5):
    ncc = self.ncc_loss(tgt, y)
    mind = self.mind_loss(tgt,y)
    grad = self.gradient_loss(flow)
    multi = self.multi_loss(src, tgt, flow1, flow2, hyper_3, hyper_4, hyper_5)
    loss = multi + hyper_1 * (hyper_5*ncc + (1-hyper_5)*mind) + hyper_2 * grad
    return loss, ncc, mind, grad


class multi_loss(nn.Module):
  def __init__(self, shape):
    super(multi_loss, self).__init__()

    inshape = shape
    down_shape2 = [int(d / 4) for d in inshape]
    down_shape1 = [int(d / 2) for d in inshape]
    self.ncc_loss = ncc_loss()
    self.mind_loss = MIND()
    self.gradient_loss = gradient_loss()
    self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
    self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
    self.resize_1 = ResizeTransform(2, len(inshape))
    self.resize_2 = ResizeTransform(4, len(inshape))

  def forward(self, src, tgt, flow1, flow2, hyper_3, hyper_4, hyper_5):
    loss = 0.
    zoomed_x1 = self.resize_1(tgt)
    zoomed_x2 = self.resize_1(src)
    warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
    loss += hyper_3 * hyper_5 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[7])+\
            hyper_3 * (1-hyper_5) * self.mind_loss(warped_zoomed_x2, zoomed_x1)

    zoomed_x1 = self.resize_2(tgt)
    zoomed_x2 = self.resize_2(src)
    warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
    loss += hyper_4 * hyper_5 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[5])+\
            hyper_4 * (1-hyper_5) * self.mind_loss(warped_zoomed_x2, zoomed_x1)

    return loss



class multi_loss_lambda(nn.Module):
  def __init__(self):
    super(multi_loss_lambda, self).__init__()

    inshape = shape
    down_shape2 = [int(d / 4) for d in inshape]
    down_shape1 = [int(d / 2) for d in inshape]
    self.ncc_loss = ncc_loss()
    self.mind_loss = MIND()
    self.gradient_loss = gradient_loss()
    self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
    self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
    self.resize_1 = ResizeTransform(2, len(inshape))
    self.resize_2 = ResizeTransform(4, len(inshape))

  def forward(self, src, tgt, flow1, flow2, hyper_1, hyper_2, hyper_3, hyper_4):
    loss = 0.
    zoomed_x1 = self.resize_1(tgt)
    zoomed_x2 = self.resize_1(src)
    warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
    loss += hyper_3 * (hyper_1 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[7]) + \
                       hyper_2 * self.mind_loss(warped_zoomed_x2, zoomed_x1))

    zoomed_x1 = self.resize_2(tgt)
    zoomed_x2 = self.resize_2(src)
    warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
    loss += hyper_4 *(hyper_1 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[5]) +  \
                      hyper_2 * self.mind_loss(warped_zoomed_x2, zoomed_x1))

    return loss


class LossFunction_mind(nn.Module):
    def __init__(self):
        super(LossFunction_mind, self).__init__()
        self.sim_loss = MIND()
        self.gradient_loss = gradient_loss()
        self.flow_jacdet_loss = jacdet_loss()
        self.multi_loss = multi_mind_loss()

    def forward(self, y, tgt, src, flow, flow_pyramid, hyper_1, hyper_2, hyper_3, hyper_4):

        flow1, refine_flow1, flow2, refine_flow2 = flow_pyramid
        ncc = self.sim_loss(tgt, y)
        grad = self.gradient_loss(flow)
        multi = self.multi_loss(src, tgt, flow1, refine_flow1, flow2, refine_flow2, hyper_3, hyper_4)
        loss = multi + hyper_1 * ncc + hyper_2 * grad
        return loss, ncc, grad


class LossFunction_ncc(nn.Module):
    def __init__(self):
        super(LossFunction_ncc, self).__init__()
        self.ncc_loss = ncc_loss()
        self.gradient_loss = gradient_loss()
        self.flow_jacdet_loss = jacdet_loss()
        self.multi_loss = multi_ncc_loss()

    def forward(self, y, tgt, src, flow, flow_pyramid, hyper_1, hyper_2, hyper_3, hyper_4):

        int_flow1, refine_flow1, int_flow2, refine_flow2 = flow_pyramid
        ncc = self.ncc_loss(tgt, y)
        grad = self.gradient_loss(flow)
        multi = self.multi_loss(src, tgt, int_flow1, refine_flow1, int_flow2, refine_flow2, hyper_3, hyper_4)
        loss = multi + hyper_1 * ncc + hyper_2 * grad 
        return loss, ncc, grad


class LossFunction_dice(nn.Module):
    def __init__(self):
        super(LossFunction_dice, self).__init__()
        self.dice_loss = Dice()
        self.gradient_loss = gradient_loss()
        self.spatial_transform = SpatialTransformer(volsize=shape) 

    def forward(self, mask_0, mask_1, flow): # num_classes=16 for ADNI num_classes=4 for Multi
        mask_0 = F.one_hot(mask_0.squeeze(1).to(torch.int64), num_classes=16).permute(0, 4, 1, 2, 3).float()  # [1, 16, 160, 192, 224]
        mask_1 = F.one_hot(mask_1.squeeze(1).to(torch.int64), num_classes=16).permute(0, 4, 1, 2, 3).float()
        mask = self.spatial_transform(mask_1, flow)
        grad = self.gradient_loss(flow)
        loss = self.dice_loss(mask_0, mask) + (grad-0.01)*(grad-0.01)*10000
        return loss


class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0


class jacdet_loss(nn.Module):
    def __init__(self):
        super(jacdet_loss, self).__init__()

    def Get_Grad(self, y):
        ndims = 3

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            # r = [d, *range(d), *range(d + 1, ndims + 2)]
            # y = K.permute_dimensions(y, r)
            y = y.permute(d, *range(d), *range(d + 1, ndims + 2))

            dfi = y[1:, ...] - y[:-1, ...]
            # [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            dfi = F.pad(dfi, pad=(0,0, 0,0, 0,0, 0,0, 1, 0), mode="constant", value=0)

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            # r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            # df[i] = K.permute_dimensions(dfi, r)
            df[i] = dfi.permute(*range(1, d + 1), 0, *range(d + 1, ndims + 2))
        # df[2] = K.permute_dimensions(df[2], (1, 0, 2, 3, 4))
        df[2] = df[2].permute(1, 0, 2, 3, 4)
        return df

    def forward(self, x):
        flow = x[0, :, :, :, :]
        vol_size = flow.shape[:-1]
        grid = np.stack(pynd.ndutils.volsize2ndgrid(vol_size), len(vol_size))
        grid = np.reshape(grid, (1,) + grid.shape)
        grid = torch.from_numpy(grid)
        J = self.Get_Grad(x + grid)
        # J = np.gradient(flow + grid)

        dx = J[0][0, :, :, :, :]
        dy = J[1][:, 0, :, :, :]
        dz = J[2][:, :, 0, :, :]

        Jdet0 = dx[:, :, :, 0] * (dy[:, :, :, 1] * dz[:, :, :, 2] - dy[:, :, :, 2] * dz[:, :, :, 1])
        Jdet1 = dx[:, :, :, 1] * (dy[:, :, :, 0] * dz[:, :, :, 2] - dy[:, :, :, 2] * dz[:, :, :, 0])
        Jdet2 = dx[:, :, :, 2] * (dy[:, :, :, 0] * dz[:, :, :, 1] - dy[:, :, :, 1] * dz[:, :, :, 0])

        Jdet = Jdet0 - Jdet1 + Jdet2

        loss = np.sum(np.maximum(0.0, -Jdet))

        return loss


class ncc_loss(nn.Module):
    def __init__(self):
        super(ncc_loss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
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

    def forward(self, I, J, win=None):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims

        conv_fn = getattr(F, 'conv%dd' % ndims)
        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -1 * torch.mean(cc)


class multi_ncc_loss(nn.Module):
  def __init__(self):
    super(multi_ncc_loss, self).__init__()

    inshape = shape
    down_shape2 = [int(d / 4) for d in inshape]
    down_shape1 = [int(d / 2) for d in inshape]
    self.ncc_loss = ncc_loss()
    self.gradient_loss = gradient_loss()
    self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
    self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
    self.resize_1 = ResizeTransform(2, len(inshape))
    self.resize_2 = ResizeTransform(4, len(inshape))

  def forward(self, src, tgt, flow1, refine_flow1, flow2, refine_flow2, hyper_3, hyper_4):
    loss = 0.
    zoomed_x1 = self.resize_1(tgt)
    zoomed_x2 = self.resize_1(src)
    warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
    loss += hyper_3 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[7])

    zoomed_x1 = self.resize_2(tgt)
    zoomed_x2 = self.resize_2(src)
    warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
    loss += hyper_4 * self.ncc_loss(warped_zoomed_x2, zoomed_x1, win=[5])

    return loss


class multi_mind_loss(nn.Module):
  def __init__(self):
    super(multi_mind_loss, self).__init__()

    inshape = shape
    down_shape2 = [int(d / 4) for d in inshape]
    down_shape1 = [int(d / 2) for d in inshape]
    self.sim_loss = MIND()
    self.gradient_loss = gradient_loss()
    self.spatial_transform_1 = SpatialTransformer(volsize=down_shape1)
    self.spatial_transform_2 = SpatialTransformer(volsize=down_shape2)
    self.resize_1 = ResizeTransform(2, len(inshape))
    self.resize_2 = ResizeTransform(4, len(inshape))

  def forward(self, src, tgt, flow1, refine_flow1, flow2, refine_flow2, hyper_3, hyper_4):
    zoomed_x1 = self.resize_1(tgt)
    zoomed_x2 = self.resize_1(src)
    warped_zoomed_x2 = self.spatial_transform_1(zoomed_x2, flow1)
    loss_1 = hyper_3 * self.sim_loss(warped_zoomed_x2, zoomed_x1)

    zoomed_x1 = self.resize_2(tgt)
    zoomed_x2 = self.resize_2(src)
    warped_zoomed_x2 = self.spatial_transform_2(zoomed_x2, flow2)
    loss_2 = hyper_4 * self.sim_loss(warped_zoomed_x2, zoomed_x1)
    loss = loss_1 + loss_2

    return loss


class Dice(nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


def add_n(l):
  res = l[0]
  for i in range(1, len(l)):
    res = torch.add(res, i)
  return res


def det3x3(M):
  M = [[M[:, i, j] for j in range(3)] for i in range(3)]
  return add_n([
    M[0][0] * M[1][1] * M[2][2],
    M[0][1] * M[1][2] * M[2][0],
    M[0][2] * M[1][0] * M[2][1]
  ]) - add_n([
    M[0][0] * M[1][2] * M[2][1],
    M[0][1] * M[1][0] * M[2][2],
    M[0][2] * M[1][1] * M[2][0]
  ])


def elem_sym_polys_of_eigen_values(M):
  M = [[M[:, i, j] for j in range(3)] for i in range(3)]
  sigma1 = add_n([M[0][0], M[1][1], M[2][2]])

  sigma2 = add_n([
    M[0][0] * M[1][1],
    M[1][1] * M[2][2],
    M[2][2] * M[0][0]
  ]) - add_n([
    M[0][1] * M[1][0],
    M[1][2] * M[2][1],
    M[2][0] * M[0][2]
  ])
  sigma3 = add_n([
    M[0][0] * M[1][1] * M[2][2],
    M[0][1] * M[1][2] * M[2][0],
    M[0][2] * M[1][0] * M[2][1]
  ]) - add_n([
    M[0][0] * M[1][2] * M[2][1],
    M[0][1] * M[1][0] * M[2][2],
    M[0][2] * M[1][1] * M[2][0]
  ])
  return sigma1, sigma2, sigma3
