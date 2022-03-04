# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************


from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal_op')

PRIMITIVES = [
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_7x7',
]

##Search_ABIDE
Search_ABIDE = Genotype (
    normal_op=['sep_conv_3x3', 'dil_conv_3x3', 'sep_conv_5x5', 'sep_conv_3x3', 'dil_conv_3x3', 'sep_conv_3x3'])