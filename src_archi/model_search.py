# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************


import torch.nn.functional as F
from operations import OPS
from torch.autograd import Variable
from genotypes import *
from layers import *

class MixedOp_2(nn.Module):
    # 将两节点各种操作实现并加入列表，输出则是对各操作的加权相加，所以输入输出维度一样
    def __init__(self, PRIMITIVES_TEST, in_channels, out_channels, stride):
        super(MixedOp_2, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_TEST:
            op = OPS[primitive](in_channels, out_channels, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        # weights就是alpha 输出是对各操作的加权相加
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell_Op(nn.Module):
    def __init__(self, PRIMITIVES_TEST, in_channels, out_channels, stride=1):
        super(Cell_Op, self).__init__()
        self.op = MixedOp_2(PRIMITIVES_TEST, in_channels, out_channels, stride)

    def forward(self, x, weights):
        out = self.op(x, weights)
        return out


class MatchCost_Cell(nn.Module):
    def __init__(self, PRIMITIVES):
        super(MatchCost_Cell, self).__init__()
        self.Primitives = PRIMITIVES

        self.enc_2 = nn.ModuleList()
        self.enc_1 = nn.ModuleList()

        self.enc_2.append(Cell_Op(self.Primitives, 33, 48, 1))  # 2层 (in_channels, out_channels, stride=1)
        self.enc_2.append(Cell_Op(self.Primitives, 48, 32, 1))  # 2层
        self.enc_2.append(Cell_Op(self.Primitives, 32, 16, 1))  # 2层

        self.enc_1.append(Cell_Op(self.Primitives, 36, 48, 1))  # 1层
        self.enc_1.append(Cell_Op(self.Primitives, 48, 32, 1))  # 1层
        self.enc_1.append(Cell_Op(self.Primitives, 32, 16, 1))  # 1层

    def forward(self, src, tgt, up_int_flow, upfeat, weights):
        mc = torch.norm(src - tgt, p=1, dim=1)
        mc = mc[..., np.newaxis]
        x = mc.permute(0, 4, 1, 2, 3)
        input_list = list(src.size())
        channel = int(input_list[1])
        if channel == 16:
            x = torch.cat((x, tgt, up_int_flow, upfeat), 1)
            x = self.enc_1[2](self.enc_1[1](self.enc_1[0](x, weights[0]), weights[1]), weights[2])
        elif channel == 32:
            x = torch.cat((tgt, x), 1)
            x = self.enc_2[2](self.enc_2[1](self.enc_2[0](x, weights[0]), weights[1]), weights[2])
        return x


class Concat_Cell(nn.Module):

    def __init__(self, PRIMITIVES):
        super(Concat_Cell, self).__init__()
        self.Primitives = PRIMITIVES

        self.enc_2 = nn.ModuleList()
        self.enc_1 = nn.ModuleList()

        self.enc_2.append(Cell_Op(self.Primitives, 64, 48, 1))  # 2层 (in_channels, out_channels, stride=1)
        self.enc_2.append(Cell_Op(self.Primitives, 48, 32, 1))  # 2层
        self.enc_2.append(Cell_Op(self.Primitives, 32, 16, 1))  # 2层

        self.enc_1.append(Cell_Op(self.Primitives, 51, 48, 1))  # 1层
        self.enc_1.append(Cell_Op(self.Primitives, 48, 32, 1))  # 1层
        self.enc_1.append(Cell_Op(self.Primitives, 32, 16, 1))  # 1层

    def forward(self, src, tgt, up_int_flow, upfeat, weights):
        input_list = list(src.size())
        channel = int(input_list[1])
        if channel == 16:
            x = torch.cat((src, tgt, up_int_flow, upfeat), 1)
            x = self.enc_1[2](self.enc_1[1](self.enc_1[0](x, weights[0]), weights[1]), weights[2])
        elif channel == 32:
            x = torch.cat((src, tgt), 1)
            x = self.enc_2[2](self.enc_2[1](self.enc_2[0](x, weights[3]), weights[4]), weights[5])
        return x


class Refine_Cell(nn.Module):

    def __init__(self, PRIMITIVES):
        super(Refine_Cell, self).__init__()
        self.Primitives = PRIMITIVES

        self.enc_2 = nn.ModuleList()
        self.enc_1 = nn.ModuleList()

        self.enc_2.append(Cell_Op(self.Primitives, 65, 48, 1))  # 2层 (in_channels, out_channels, stride=1)
        self.enc_2.append(Cell_Op(self.Primitives, 48, 32, 1))  # 2层
        self.enc_2.append(Cell_Op(self.Primitives, 32, 16, 1))  # 2层

        self.enc_1.append(Cell_Op(self.Primitives, 52, 48, 1))  # 1层
        self.enc_1.append(Cell_Op(self.Primitives, 48, 32, 1))  # 1层
        self.enc_1.append(Cell_Op(self.Primitives, 32, 16, 1))  # 1层

    def forward(self, src, tgt, up_int_flow, upfeat, weights):
        mc = torch.norm(src - tgt, p=1, dim=1)
        mc = mc[..., np.newaxis]
        x = mc.permute(0, 4, 1, 2, 3)
        input_list = list(src.size())
        channel = int(input_list[1])
        if channel == 16:
            x = torch.cat((src, tgt, x, up_int_flow, upfeat), 1)
            x = self.enc_1[2](self.enc_1[1](self.enc_1[0](x, weights[0]), weights[1]), weights[2])
        elif channel == 32:
            x = torch.cat((src, tgt, x), 1)
            x = self.enc_2[2](self.enc_2[1](self.enc_2[0](x, weights[0]), weights[1]), weights[2])
        return x


Cell_Operation = {
    'MatchCost_Cell': lambda genotype: MatchCost_Cell(PRIMITIVES),
    'Concat_Cell': lambda genotype: Concat_Cell(PRIMITIVES),
    'Refine_Cell': lambda genotype: Refine_Cell(PRIMITIVES),
}


class Network(nn.Module):
    def __init__(self, criterion, cell_name, shape):
        super(Network, self).__init__()
        self._criterion = criterion
        self._cellname = cell_name
        self.cell = Cell_Operation[cell_name](PRIMITIVES)
        self._initialize_alphas()

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
        self.fea = FeatureExtraction_16_search()

    def forward(self, src, tgt):
        s0, t0, s1, t1 = self.fea(src, tgt)
        up_int_flow = None
        up_feat = None
        weights_alpha = F.softmax(self.alphas, dim=-1)
        f2 = self.cell(s1, t1, up_int_flow, up_feat, weights_alpha)
        flow2 = self.predict_flow(f2)
        upfeat2 = self.resize(f2)
        int_flow2 = self.integrate2(flow2)
        up_int_flow2 = self.resize(int_flow2)
        features_s_warped = self.spatial_transform_f(s0, up_int_flow2)

        f1 = self.cell(features_s_warped, t0, up_int_flow2, upfeat2, weights_alpha)
        flow1 = self.predict_flow(f1)
        int_flow1 = self.integrate1(flow1)

        flow = self.resize(int_flow1)
        flow = self.integrate0(flow)
        y = self.spatial_transform(src, flow)
        return y, flow, int_flow1, int_flow2

    def _loss(self, src, tgt):
        y, flow, flow1, flow2 = self.forward(src, tgt)
        return self._criterion(y, tgt, src, flow, flow1, flow2)

    def new(self):
        model_new = Network(self._criterion, self._cellname).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_alphas(self):

        num_ops = len(PRIMITIVES)
        k_concat = 6
        self.alphas = Variable(1e-3 * torch.randn(k_concat, num_ops).cuda(), requires_grad=True)#Variable 没看懂
        self._arch_parameters = [
            self.alphas,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights, k=6):
            gene = []
            for i in range(k):
                W = weights[i].copy()
                k_best = None
                for k in range(len(W)):
                    if k_best is None or W[k] > W[k_best]:
                        k_best = k
                gene.append((PRIMITIVES[k_best]))
            return gene

        gene = _parse(F.softmax(self.alphas, dim=-1).data.cpu().numpy())
        genotype = Genotype(normal_op=gene)
        return genotype
