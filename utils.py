# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************

import os
import sys
import numpy as np
import torch
import shutil
import nibabel as nib

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1,1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


def mask_metrics(seg1, seg2):
    ''' Given two segmentation seg1, seg2, 0 for background 255 for foreground.
    Calculate the Dice score
    $ 2 * | seg1 \cap seg2 | / (|seg1| + |seg2|) $
    and the Jacc score
    $ | seg1 \cap seg2 | / (|seg1 \cup seg2|) $
    '''
    seg1 = seg1 > 128
    seg2 = seg2 > 128
    seg1 = seg1.astype(np.float32)
    seg2 = seg2.astype(np.float32)
    dice_score = 2.0 * np.sum(seg1 * seg2) / (
        np.sum(seg1) + np.sum(seg2))
    union = np.sum(np.maximum(seg1, seg2))
    return (dice_score, np.sum(np.minimum(seg1, seg2)) / np.maximum(0.01, union))


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def print_network(net):
  num_params = 0
  for param in net.parameters():
    num_params += param.numel()
  print('Total number of parameters: %d' % num_params)


def load_nii(vol_name):
    X = nib.load(vol_name).get_data()
    # X = np.reshape(X, X.shape + (1,))
    return X


def load_nii_by_name(vol_name, seg_name):
    X = nib.load(vol_name).get_data()
    X = np.reshape(X, X.shape + (1,))
    return_vals = [X]

    X_seg = nib.load(seg_name).get_data()
    return_vals.append(X_seg)

    return tuple(return_vals)


def load_nii_by_name_norm(vol_name, seg_name):
    X = nib.load(vol_name).get_data()
    X = X / np.max(X)
    X = np.reshape(X, X.shape + (1,))
    return_vals = [X]

    X_seg = nib.load(seg_name).get_data()
    return_vals.append(X_seg)

    return tuple(return_vals)

