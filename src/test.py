# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************

import os
import sys
import nibabel as nib
base_dir = (os.path.dirname(os.path.dirname(os.path.abspath((__file__)))))
sys.path.append(base_dir)

from argparse import ArgumentParser
from model import *
from utils import dice
from dataset import load_volfile

def test(gpu, data_dir, model_file):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param model_file: the model directory to load from
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Produce the loaded atlas with dims.:160x192x224.
    train_vol_names = np.load(data_dir)['arr_0']

    # # Anatomical labels we want to evaluate
    # good_labels = sio.loadmat('../data/labels.mat')['labels'][0]

    # Set up model
    model = MPRNet().cuda()
    model.to(device)
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))


    # Use this to warp segments
    trf = SpatialTransformer(shape, mode='nearest').to(device)
    val_affine_list, val_list = [], []
    for i in range(0, len(train_vol_names)):

        Target_image, Source_image, seg_path_0, seg_path_1 = train_vol_names[i]

        image_0 = load_volfile(Target_image)
        image_0 = torch.Tensor(image_0).float()
        image_0 = image_0.unsqueeze(0).unsqueeze(0).cuda()

        image_1 = load_volfile(Source_image)
        image_1 = torch.Tensor(image_1).float()
        image_1 = image_1.unsqueeze(0).unsqueeze(0).cuda()

        mask_0 = load_volfile(seg_path_0)
        mask_1 = load_volfile(seg_path_1)

        mask_1 = torch.Tensor(mask_1).float()
        mask_1 = mask_1.unsqueeze(0).unsqueeze(0).cuda()

        y, flow, flow_pyramid = model(image_0, image_1)

        # Warp segment using flow
        warp_seg = trf(mask_1, flow).detach().cpu().numpy()
        mask_1 = mask_1.detach().cpu().numpy()

        val = dice(mask_1[0,0,:,:,:], mask_0)
        print('affine:', np.mean(val))
        val_affine_list.append(np.mean(val))

        val = dice(warp_seg[0,0,:,:,:], mask_0)
        print('registered:', np.mean(val))
        val_list.append(np.mean(val))
        
        # Visualize the warped segment
        hdr = nib.load(Source_image).header
        array = hdr.get_qform(coded=False)
        warped_image = nib.Nifti1Image(warp_seg.squeeze(), array)
        nib.save(warped_image, os.path.join(save_data_dir, Source_image.split('\')[-1])


    print("affine-mean:", np.mean(val_affine_list))
    print("affine-std:", np.std(val_affine_list))

    print("registered-mean:", np.mean(val_list))
    print("registered-std:", np.std(val_list))

    np.savez('results/9-T2toT1_test_Dice.npz', val_list)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help="gpu id")
    parser.add_argument("--data_dir", type=str, default='data/9-T2toT1_test_data_with_mask.npz')
    parser.add_argument("--model_file", type=str, default='MPRNet-T2-to-T1-singlemodel/399.ckpt')
    parser.add_argument("--save_data_dir", type=str, default='data/warped_image')
    test(**vars(parser.parse_args()))

