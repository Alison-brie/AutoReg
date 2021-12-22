# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************

import numpy as np
import nibabel as nib
import scipy.io as sio

image_path_table = np.load('/root/data/registration/MultiPropReg/data/test_data_with_mask.npz')['arr_0']
good_labels = sio.loadmat('/root/data/registration/MultiPropReg/data/labels.mat')['labels'][0][:15]
index = list(range(len(good_labels)))

def convert(seg):
    labels = (np.unique(seg))
    extra_labels = list(set(labels).difference(set(good_labels)))

    output = np.copy(seg)
    for i in extra_labels:
        output[seg == i] = 15
    for k, v in zip(good_labels, index):
        output[seg == k ] = v
    return output



