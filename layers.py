# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************


import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# shape = (160, 192, 224)
shape = (160, 160, 160)


class FeatureLearning(nn.Module):

    def __init__(self):

        super(FeatureLearning, self).__init__()

        # FeatureLearning/Encoder functions
        dim = 3
        self.enc = nn.ModuleList()
        self.enc.append(conv_block(dim, 1, 16, 2))  # 0 (dim, in_channels, out_channels, stride=1)
        self.enc.append(conv_block(dim, 16, 16, 1)) # 1
        self.enc.append(conv_block(dim, 16, 16, 1)) # 2
        self.enc.append(conv_block(dim, 16, 32, 2)) # 3
        self.enc.append(conv_block(dim, 32, 32, 1)) # 4
        self.enc.append(conv_block(dim, 32, 32, 1)) # 5

    def forward(self, src, tgt):

        c11 = self.enc[2](self.enc[1](self.enc[0](src)))
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt)))
        c12 = self.enc[5](self.enc[4](self.enc[3](c11)))
        c22 = self.enc[5](self.enc[4](self.enc[3](c21)))

        return c11, c21, c12, c22


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize= shape , mode='bilinear'): # (160, 192, 224)
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src_loss and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out


def predict_flow(in_planes):
    dim = 3
    conv_fn = getattr(nn, 'Conv%dd' % dim)
    return conv_fn(in_planes, dim, kernel_size=3, padding=1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def MatchCost(features_t, features_s):
    mc = torch.norm(features_t - features_s, p=1, dim=1)
    mc = mc[ ..., np.newaxis]
    return mc.permute(0, 4, 1, 2, 3)



#########################################################################################################


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.LeakyReLU(0.2))
    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.LeakyReLU(0.2))
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


Genotype = namedtuple('Genotype', 'normal_op')
genotype = Genotype(normal_op=['sep_conv_3x3', 'conv_5x5', 'conv_3x3', 'dil_conv_7x7', 'dil_conv_7x7', 'sep_conv_5x5'])

OPS = {
    'conv_1x1': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 1, stride, 0),
    'conv_3x3': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 3, stride, 1),
    'conv_5x5': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, 5, stride, 2),
    'sep_conv_3x3': lambda C_in, C_out, stride: SepConv(C_in, C_out, 3, stride, 1),
    'sep_conv_5x5': lambda C_in, C_out, stride: SepConv(C_in, C_out, 5, stride, 2),
    'dil_conv_3x3': lambda C_in, C_out, stride: DilConv(C_in, C_out, 3, stride, 2, 2),
    'dil_conv_5x5': lambda C_in, C_out, stride: DilConv(C_in, C_out, 5, stride, 4, 2),
    'dil_conv_7x7': lambda C_in, C_out, stride: DilConv(C_in, C_out, 7, stride, 6, 2),
    'dil_conv_3x3_8': lambda C_in, C_out, stride: DilConv(C_in, C_out, 3, stride, 8, 8), }


class ModelCell(nn.Module):
  def __init__(self, primitive, in_channels, out_channels, stride=1):
    super(ModelCell, self).__init__()

    self.op = OPS[primitive](in_channels, out_channels, stride)

  def forward(self, x):
    return self.op(x)


class FeatureExtraction_16(nn.Module):


    def __init__(self):
        super(FeatureExtraction_16, self).__init__()
        #将搜索出来的特征提取器固定
        # self.genotype = Encoder_V3
        self.operations = ['conv_3x3', 'sep_conv_5x5', 'conv_5x5', 'sep_conv_5x5', 'dil_conv_3x3_8', 'dil_conv_3x3_8']
        #PPMI:'conv_3x3', 'sep_conv_5x5', 'conv_5x5', 'sep_conv_5x5', 'dil_conv_3x3_8', 'dil_conv_3x3_8'
        # 'conv_3x3', 'conv_3x3', 'conv_3x3', 'conv_3x3', 'conv_3x3', 'conv_3x3'

        self.enc = nn.ModuleList()
        self.enc.append(ModelCell(self.operations[0], 1, 16, 2))  # 0 (in_channels, out_channels, stride=1)
        self.enc.append(ModelCell(self.operations[1], 16, 16, 1))  # 1
        self.enc.append(ModelCell(self.operations[2], 16, 16, 1))  # 2
        self.enc.append(ModelCell(self.operations[3], 16, 32, 2))  # 3
        self.enc.append(ModelCell(self.operations[4], 32, 32, 1))  # 4
        self.enc.append(ModelCell(self.operations[5], 32, 32, 1))  # 5

    def forward(self, src, tgt):
        c11 = self.enc[2](self.enc[1](self.enc[0](src)))
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt)))
        c12 = self.enc[5](self.enc[4](self.enc[3](c11)))
        c22 = self.enc[5](self.enc[4](self.enc[3](c21)))
        s0, t0, s1, t1 = c11, c21, c12, c22
        return s0, t0, s1, t1


Cell_Operation = {'Concat_Cell': lambda genotype: Concat_Cell(genotype)}


class Concat_Cell(nn.Module):

    def __init__(self, genotype):
        super(Concat_Cell, self).__init__()
        self.genotype = genotype.normal_op

        self.enc_2 = nn.ModuleList()
        self.enc_1 = nn.ModuleList()

        self.enc_2.append(ModelCell(self.genotype[3], 64, 48, 1))  # 2层 (in_channels, out_channels, stride=1)
        self.enc_2.append(ModelCell(self.genotype[4], 48, 32, 1))  # 2层
        self.enc_2.append(ModelCell(self.genotype[5], 32, 16, 1))  # 2层

        self.enc_1.append(ModelCell(self.genotype[0], 51, 48, 1))  # 1层
        self.enc_1.append(ModelCell(self.genotype[1], 48, 32, 1))  # 1层
        self.enc_1.append(ModelCell(self.genotype[2], 32, 16, 1))  # 1层

    def forward(self, src, tgt, up_int_flow, upfeat):

        input_list = list(src.size())
        channel = int(input_list[1])
        if channel == 16:
            x = torch.cat((src, tgt, up_int_flow, upfeat), 1)
            x = self.enc_1[2](self.enc_1[1](self.enc_1[0](x)))
        elif channel == 32:
            x = torch.cat((src, tgt), 1)
            x = self.enc_2[2](self.enc_2[1](self.enc_2[0](x)))
        return x



class FeatureExtraction_16_search(nn.Module):
    def __init__(self):
        super(FeatureExtraction_16_search, self).__init__()
        #将搜索出来的特征提取器固定
        self.operations = ['conv_3x3', 'sep_conv_5x5', 'conv_5x5', 'sep_conv_5x5', 'dil_conv_3x3_8', 'dil_conv_3x3_8']
        self.enc = nn.ModuleList()
        self.enc.append(ModelCell(self.operations[0], 1, 16, 2))  # 0 (in_channels, out_channels, stride=1)
        self.enc.append(ModelCell(self.operations[1], 16, 16, 1))  # 1
        self.enc.append(ModelCell(self.operations[2], 16, 16, 1))  # 2
        self.enc.append(ModelCell(self.operations[3], 16, 32, 2))  # 3
        self.enc.append(ModelCell(self.operations[4], 32, 32, 1))  # 4
        self.enc.append(ModelCell(self.operations[5], 32, 32, 1))  # 5

    def forward(self, src, tgt):
        c11 = self.enc[2](self.enc[1](self.enc[0](src)))
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt)))
        c12 = self.enc[5](self.enc[4](self.enc[3](c11)))
        c22 = self.enc[5](self.enc[4](self.enc[3](c21)))
        s0, t0, s1, t1 = c11, c21, c12, c22
        return s0, t0, s1, t1


class FeatureExtraction_16_train(nn.Module):
    def __init__(self):
        super(FeatureExtraction_16_train, self).__init__()
        #将搜索出来的特征提取器固定
        # self.genotype = Encoder_V3
        self.operations = ['conv_3x3', 'sep_conv_5x5', 'conv_5x5', 'sep_conv_5x5', 'dil_conv_3x3_8', 'dil_conv_3x3_8']

        self.enc = nn.ModuleList()
        self.enc.append(ModelCell(self.operations[0], 1, 16, 2))  # 0 (in_channels, out_channels, stride=1)
        self.enc.append(ModelCell(self.operations[1], 16, 16, 1))  # 1
        self.enc.append(ModelCell(self.operations[2], 16, 16, 1))  # 2
        self.enc.append(ModelCell(self.operations[3], 16, 32, 2))  # 3
        self.enc.append(ModelCell(self.operations[4], 32, 32, 1))  # 4
        self.enc.append(ModelCell(self.operations[5], 32, 32, 1))  # 5

    def forward(self, src, tgt):
        c11 = self.enc[2](self.enc[1](self.enc[0](src)))
        c21 = self.enc[2](self.enc[1](self.enc[0](tgt)))
        c12 = self.enc[5](self.enc[4](self.enc[3](c11)))
        c22 = self.enc[5](self.enc[4](self.enc[3](c21)))
        s0, t0, s1, t1 = c11, c21, c12, c22
        return s0, t0, s1, t1

