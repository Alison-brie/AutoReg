# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************


from layers import *

Cell_Operation = {
    'MatchCost_Cell': lambda genotype: MatchCost_1_Cell(genotype),
    'Concat_Cell': lambda genotype: Concat_Cell(genotype),
    'Refine_Cell': lambda genotype: Refine_Cell(genotype),
}


class ModelCell(nn.Module):
    def __init__(self, primitive, in_channels, out_channels, stride=1):
        super(ModelCell, self).__init__()

        self.op = OPS[primitive](in_channels, out_channels, stride)

    def forward(self, x):
        return self.op(x)


class MatchCost_1_Cell(nn.Module):
    def __init__(self, genotype):
        super(MatchCost_1_Cell, self).__init__()
        self.genotype = genotype.normal_mc

        self.enc_2 = nn.ModuleList()
        self.enc_1 = nn.ModuleList()

        self.enc_2.append(ModelCell(self.genotype[0], 33, 48, 1))  # 2层 (in_channels, out_channels, stride=1)
        self.enc_2.append(ModelCell(self.genotype[1], 48, 32, 1))  # 2层
        self.enc_2.append(ModelCell(self.genotype[2], 32, 16, 1))  # 2层

        self.enc_1.append(ModelCell(self.genotype[0], 36, 48, 1))  # 1层
        self.enc_1.append(ModelCell(self.genotype[1], 48, 32, 1))  # 1层
        self.enc_1.append(ModelCell(self.genotype[2], 32, 16, 1))  # 1层

    def forward(self, src, tgt, up_int_flow, upfeat):
        mc = torch.norm(src - tgt, p=1, dim=1)
        mc = mc[..., np.newaxis]
        x = mc.permute(0, 4, 1, 2, 3)
        input_list = list(src.size())
        channel = int(input_list[1])
        if channel == 16:
            x = torch.cat((x, tgt, up_int_flow, upfeat), 1)
            x = self.enc_1[2](self.enc_1[1](self.enc_1[0](x)))
        elif channel == 32:
            x = torch.cat((tgt, x), 1)
            x = self.enc_2[2](self.enc_2[1](self.enc_2[0](x)))
        return x


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


class Refine_Cell(nn.Module):

    def __init__(self, genotype):
        super(Refine_Cell, self).__init__()
        self.genotype = genotype.normal_op

        self.enc_2 = nn.ModuleList()
        self.enc_1 = nn.ModuleList()

        self.enc_2.append(ModelCell(self.genotype[0], 65, 48, 1))  # 2层 (in_channels, out_channels, stride=1)
        self.enc_2.append(ModelCell(self.genotype[1], 48, 32, 1))  # 2层
        self.enc_2.append(ModelCell(self.genotype[2], 32, 16, 1))  # 2层

        self.enc_1.append(ModelCell(self.genotype[0], 52, 48, 1))  # 1层
        self.enc_1.append(ModelCell(self.genotype[1], 48, 32, 1))  # 1层
        self.enc_1.append(ModelCell(self.genotype[2], 32, 16, 1))  # 1层

    def forward(self, src, tgt, up_int_flow, upfeat):
        mc = torch.norm(src - tgt, p=1, dim=1)
        mc = mc[..., np.newaxis]
        x = mc.permute(0, 4, 1, 2, 3)
        input_list = list(src.size())
        channel = int(input_list[1])
        if channel == 16:
            x = torch.cat((src, tgt, x, up_int_flow, upfeat), 1)
            x = self.enc_1[2](self.enc_1[1](self.enc_1[0](x)))
        elif channel == 32:
            x = torch.cat((src, tgt, x), 1)
            x = self.enc_2[2](self.enc_2[1](self.enc_2[0](x)))
        return x


class Network(nn.Module):
    def __init__(self, criterion, cell_name, genotype, shape):
        super(Network, self).__init__()
        self._criterion = criterion
        self._cellname = cell_name
        self.cell = Cell_Operation[cell_name](genotype)

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
        self.fea = FeatureExtraction_16_train()

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
        return y, flow, int_flow1, int_flow2

    def _loss(self, src, tgt):
        y, flow, flow1, flow2 = self.forward(src, tgt)
        return self._criterion(y, tgt, src, flow, flow1, flow2)
