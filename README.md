# Automated Learning for Deformable Medical Image Registration by Jointly Optimizing Network Architectures and Objective Functions

This is the official code for "Automated Learning for Deformable Medical Image Registration by Jointly Optimizing Network Architectures and Objective Functions" (IEEE TIP 2023).


A successful registration algorithm, either derived from conventional energy optimization or deep networks requires tremendous efforts from computer experts
to well design registration energy or to carefully tune network architectures for the specific type of medical data. To tackle the
aforementioned problems, this paper proposes an automated learning registration algorithm (AutoReg) that cooperatively optimizes both architectures and their corresponding training objectives, enable non-computer experts, e.g., medical/clinical users, to conveniently find off-the-shelf registration algorithms for diverse scenarios. 

<div align=center>
<img src=png/pipline-1.png width=100% />
</div>

## Prerequisites
- `Python 3.6.8+`
- `Pytorch 0.3.1`
- `torchvision 0.2.0`
- `NumPy`
- `NiBabel`

## (Example) Training on the preprocessed OASIS dataset
If you want to train on the preprocessed OASIS dataset in https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md. We have an example showing how to train on this dataset.
Download the preprocessed OASIS dataset, unzip it and put it in "Data/OASIS".


## Publication
If you find this repository useful, please cite:

- **Automated Learning for Deformable Medical Image Registration by Jointly Optimizing Network Architectures and Objective Functions**  
Fan, Xin and [Li, Zi](https://alison-brie.github.io/) and Li, Ziyang and Wang, Xiaolin and Liu, Risheng and Luo, Zhongxuan and Huang, Hao.
 IEEE TIP [eprint arXiv:2203.06810](https://arxiv.org/pdf/2203.06810)

## Keywords
Automated learning, medical Image Registration, hyperparameter learning, convolutional neural networks
