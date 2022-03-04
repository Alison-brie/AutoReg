# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************


import os
import sys
import glob
import utils
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from model_search import *
from architect_v5 import Architect
import dataset
import random
import torch.utils.data as da
from torch.optim import Adam
from losses import LossFunction_mpr

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='E:/Lzy/data/ABIDE/train', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=45, help='num of training epochs')

parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='models/1-search', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for arch encoding')
parser.add_argument('--shape', type=list, default=[120, 120, 120], help='weight decay for arch encoding')
args = parser.parse_args()

# prepare model folder
model_dir = args.save
os.makedirs(model_dir, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log_ABIDE.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = LossFunction_mpr(args.shape).cuda()
    model = Network(criterion, 'Concat_Cell', args.shape)

    # model.load_state_dict(torch.load('models/3-liver_search/14.ckpt'))
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # data generator
    train_vol_names = glob.glob(os.path.join(args.data, '*.nii.gz'))
    random.shuffle(train_vol_names)
    train_data = datagenerators.MRIDataset(train_vol_names, atlas_file='E:/Lzy/NAS-0512/data/atlas_norm.npz')
    num_train = len(train_vol_names)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = da.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=da.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=1)

    valid_queue = da.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=da.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=1)

    architect = Architect(model, args)# 计算alpha的梯度

    for epoch in range(args.epochs):
        lr = args.learning_rate
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        logging.info(F.softmax(model.alphas, dim=-1))

        # training
        loss = train(train_queue, valid_queue, model, architect, optimizer, lr, epoch)
        logging.info('train_loss of %03d epoch : %f', epoch, loss.item())

        tmp = epoch
        save_file_name = os.path.join(args.save, '%d.ckpt' % tmp)
        torch.save(model.state_dict(), save_file_name)


def train(train_queue, valid_queue, model, architect, optimizer, lr, epoch):
    for step, (input, target) in enumerate(train_queue):
        model.train()
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        if epoch >= 15:
            input_search, target_search = next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda()
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        loss, ncc, mind, grad = model._loss(input, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.report_freq == 0:
            logging.info('train %03d  %f, %f, %f, %f', step, loss, ncc, mind, grad)

    return loss


if __name__ == '__main__':
    main()
