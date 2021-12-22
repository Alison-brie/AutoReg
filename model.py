# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************

import torch.nn as nn
from layers import *


class MPRNet(nn.Module):

    def __init__(self):
        super(MPRNet, self).__init__()
        dim = 3
        int_steps = 7
        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.FeatureLearning = FeatureLearning()
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform = SpatialTransformer()

        od = 32 + 1
        self.conv2_0 = conv_block(dim, od, 48, 1)
        self.conv2_1 = conv_block(dim, 48, 32, 1)
        self.conv2_2 = conv_block(dim, 32, 16, 1)
        self.predict_flow2a = predict_flow(16)

        self.dc_conv2_0 = conv(3, 48, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2_1 = conv(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv2_2 = conv(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.predict_flow2b = predict_flow(32)

        od = 1 + 16 + 16 + 3
        self.conv1_0 = conv_block(dim, od, 48, 1)
        self.conv1_1 = conv_block(dim, 48, 32, 1)
        self.conv1_2 = conv_block(dim, 32, 16, 1)
        self.predict_flow1a = predict_flow(16)

        self.dc_conv1_0 = conv(3, 48, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv1_1 = conv(48, 48, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv1_2 = conv(48, 32, kernel_size=3, stride=1, padding=4, dilation=4)
        self.predict_flow1b = predict_flow(32)

        self.resize = ResizeTransform(1 / 2, dim)
        self.integrate2 = VecInt(down_shape2, int_steps)
        self.integrate1 = VecInt(down_shape1, int_steps)

    def forward(self, tgt, src):
        c11, c21, c12, c22 = self.FeatureLearning(tgt, src)

        corr2 = MatchCost(c22, c12)
        x = torch.cat((corr2, c22), 1)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        flow2 = self.predict_flow2a(x)
        upfeat2 = self.resize(x)

        x = self.dc_conv2_0(flow2)
        x = self.dc_conv2_1(x)
        x = self.dc_conv2_2(x)
        refine_flow2 = self.predict_flow2b(x) + flow2
        int_flow2 = self.integrate2(refine_flow2)
        up_int_flow2 = self.resize(int_flow2)
        features_s_warped = self.spatial_transform_f(c11, up_int_flow2)

        corr1 = MatchCost(c21, features_s_warped)
        x = torch.cat((corr1, c21, up_int_flow2, upfeat2), 1)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        flow1 = self.predict_flow1a(x) + up_int_flow2

        x = self.dc_conv1_0(flow1)
        x = self.dc_conv1_1(x)
        x = self.dc_conv1_2(x)
        refine_flow1 = self.predict_flow1b(x) + flow1
        int_flow1 = self.integrate1(refine_flow1)
        flow = self.resize(int_flow1)
        y = self.spatial_transform(src, flow)
        flow_pyramid = [int_flow1, refine_flow1, int_flow2, refine_flow2]
        return y, flow, flow_pyramid


class MPRNet_ST(nn.Module):

    def __init__(self, criterion_reg, criterion_seg):
        super(MPRNet_ST, self).__init__()
        self.criterion_reg, self.criterion_seg = criterion_reg, criterion_seg

        self.Reg = MPRNet()
        self.hyper_1 = torch.nn.Parameter(torch.FloatTensor([10]), requires_grad=True) 
        self.hyper_2 = torch.nn.Parameter(torch.FloatTensor([15]), requires_grad=True)
        self.hyper_3 = torch.nn.Parameter(torch.FloatTensor([3.2]), requires_grad=True)
        self.hyper_4 = torch.nn.Parameter(torch.FloatTensor([0.8]), requires_grad=True)

        # # loss = hyper_1 * ncc + hyper_2 * mind + multi + hyper_5 * grad 
        # self.hyper_1 = torch.nn.Parameter(torch.FloatTensor([10]), requires_grad=True) 
        # self.hyper_2 = torch.nn.Parameter(torch.FloatTensor([10]), requires_grad=True)
        # self.hyper_3 = torch.nn.Parameter(torch.FloatTensor([5]), requires_grad=True)
        # self.hyper_4 = torch.nn.Parameter(torch.FloatTensor([2.5]), requires_grad=True)
        # self.hyper_5 = torch.nn.Parameter(torch.FloatTensor([10]), requires_grad=True)

    def forward(self, tgt, src):
        y, flow, flow_pyramid = self.Reg(tgt, src)
        return y, flow, flow_pyramid

    def upper_parameters(self):
        return [self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4]

    def lower_parameters(self):
        return self.Reg.parameters()

    def _upper_loss(self, tgt, src, tgt_mask, src_mask):
        y, flow, flow_pyramid = self(tgt, src)
        loss = self.criterion_seg(tgt_mask, src_mask, flow)
        return loss

    def _lower_loss(self, tgt, src):
        y, flow, flow_pyramid = self(tgt, src) 
        loss = self.criterion_reg(y, tgt, src, flow, flow_pyramid, self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4)
        return loss

    def new(self):
        model_new = MPRNet_ST(self.criterion_reg, self.criterion_seg).cuda()
        for x, y in zip(model_new.upper_parameters(), self.upper_parameters()):
            x.data.copy_(y.data)
        return model_new



#########################################################################################################


class PPMINet(nn.Module):
    def __init__(self):
        super(PPMINet, self).__init__()
        self.cell = Cell_Operation['Concat_Cell'](genotype)
        int_steps = 7
        inshape = shape
        down_shape2 = [int(d / 4) for d in inshape]
        down_shape1 = [int(d / 2) for d in inshape]
        self.resize = ResizeTransform(1 / 2, ndims=3)
        self.spatial_transform_f = SpatialTransformer(volsize=down_shape1)
        self.spatial_transform = SpatialTransformer(volsize=inshape)
        self.predict_flow = predict_flow(16)
        self.integrate2 = VecInt(down_shape2, int_steps)
        self.integrate1 = VecInt(down_shape1, int_steps)
        self.integrate0 = VecInt(inshape, int_steps)
        self.fea = FeatureExtraction_16()

    def forward(self, src, tgt):
        s0, t0, s1, t1 = self.fea(src, tgt)
        up_int_flow = None
        up_feat = None

        f2 = self.cell(s1, t1, up_int_flow, up_feat)
        flow2 = self.predict_flow(f2)
        upfeat2 = self.resize(f2)

        int_flow2 = self.integrate2(flow2)
        up_int_flow2 = self.resize(int_flow2)
        features_s_warped = self.spatial_transform_f(s0, up_int_flow2)

        f1 = self.cell(features_s_warped, t0, up_int_flow2, upfeat2)
        flow1 = self.predict_flow(f1)
        int_flow1 = self.integrate1(flow1)

        flow = self.resize(int_flow1)
        flow = self.integrate0(flow)
        y = self.spatial_transform(src, flow)
        flow_pyramid = [int_flow1, int_flow1, int_flow2, int_flow1]
        return y, flow, flow_pyramid


    def _loss(self, src, tgt):
        y, flow, flow1, flow2 = self.forward(src, tgt)
        return self._criterion(y, tgt, src, flow, flow1, flow2)


class PPMINet_ST(nn.Module):

    def __init__(self, criterion_reg, criterion_seg):
        super(PPMINet_ST, self).__init__()
        self.criterion_reg, self.criterion_seg = criterion_reg, criterion_seg

        self.Reg = PPMINet()
        self.hyper_1 = torch.nn.Parameter(torch.FloatTensor([10]), requires_grad=True)
        self.hyper_2 = torch.nn.Parameter(torch.FloatTensor([15]), requires_grad=True)
        self.hyper_3 = torch.nn.Parameter(torch.FloatTensor([3.2]), requires_grad=True)
        self.hyper_4 = torch.nn.Parameter(torch.FloatTensor([0.8]), requires_grad=True)

        # # 初始化
        # self.hyper_1.data.fill_(10)
        # self.hyper_2.data.fill_(15)
        # self.hyper_3.data.fill_(3.2)
        # self.hyper_4.data.fill_(0.8)

        

    def forward(self, tgt, src):
        y, flow, flow_pyramid = self.Reg(tgt, src)
        return y, flow, flow_pyramid

    def upper_parameters(self):
        return [self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4]

    def lower_parameters(self):
        return self.Reg.parameters()

    def _upper_loss(self, tgt, src, tgt_mask, src_mask):
        y, flow, flow_pyramid = self(tgt, src)
        loss = self.criterion_seg(tgt_mask, src_mask, flow)
        return loss

    def _lower_loss(self, tgt, src):
        y, flow, flow_pyramid = self(tgt, src) 
        loss = self.criterion_reg(y, tgt, src, flow, flow_pyramid, self.hyper_1, self.hyper_2, self.hyper_3, self.hyper_4)
        return loss

    def new(self):
        model_new = PPMINet_ST(self.criterion_reg, self.criterion_seg).cuda()
        for x, y in zip(model_new.upper_parameters(), self.upper_parameters()):
            x.data.copy_(y.data)
        return model_new


