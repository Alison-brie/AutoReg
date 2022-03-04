# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************


import torch
import torch.nn as nn

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool3d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool3d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_1x1': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 1, stride, 0),
    'conv_3x3': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 3, stride, 1),
    'conv_5x5': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 5, stride, 2),
    'normal_conv': lambda C_in, C_out, stride: NormalConv(C_in, C_out, 3, stride, 1),
    'sep_conv_1x1': lambda C_in, C_out, stride: SepConv(C_in, C_out, 1, stride, 0),
    'sep_conv_3x3': lambda C_in, C_out, stride: SepConv(C_in, C_out, 3, stride, 1),
    'sep_conv_5x5': lambda C_in, C_out, stride: SepConv(C_in, C_out, 5, stride, 2),
    'sep_conv_7x7': lambda C_in, C_out, stride: SepConv(C_in, C_out, 7, stride, 3),
    'dil_conv_3x3': lambda C_in, C_out, stride: DilConv(C_in, C_out, 3, stride, 2, 2),
    'dil_conv_3x3_4': lambda C_in, C_out, stride: DilConv(C_in, C_out, 3, stride, 4, 4),
    'dil_conv_3x3_6': lambda C_in, C_out, stride: DilConv(C_in, C_out, 3, stride, 6, 6),
    'dil_conv_3x3_8': lambda C_in, C_out, stride: DilConv(C_in, C_out, 3, stride, 8, 8),
    'dil_conv_5x5': lambda C_in, C_out, stride: DilConv(C_in, C_out, 5, stride, 4, 2),
    'dil_conv_7x7': lambda C_in, C_out, stride: DilConv(C_in, C_out, 7, stride, 6, 2),
}

class NormalConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(NormalConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.op(x)

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_in, kernel_size, stride=1, padding=padding, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.op(x)



class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return out

