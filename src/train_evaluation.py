# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************

import os
import sys
base_dir = (os.path.dirname(os.path.dirname(os.path.abspath((__file__)))))
sys.path.append(base_dir)

import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import *
from dataset import MRIDataset
import random
import torch.utils.data as da
from torch.optim import Adam
from losses import LossFunction_mind, LossFunction_ncc
from torch.utils.tensorboard import SummaryWriter
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=base_dir +'/data/KneetoKnee_train_data.npz',
                    help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id') 
parser.add_argument('--save', type=str, default='MPRNet-Knee-to-Knee-singlemodel', help='experiment name')  ###
parser.add_argument('--epochs', type=int, default=400, help='num of training epochs')  ###
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')  
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--train_portion', type=float, default=1.0, help='portion of training data')
args = parser.parse_args()



if not os.path.exists(args.save):
    os.mkdir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(log_dir=args.save, flush_secs=30)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %d' % args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    criterion = LossFunction_ncc().cuda()
    model = MPRNet().cuda()
    # model.load_state_dict(torch.load('MPRNet-T2-to-T2-singlemodel/199-v1.ckpt'))
    print(utils.count_parameters_in_MB(model))
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # data generator
    train_vol_names = np.load(args.data_dir)['arr_0']
    random.shuffle(train_vol_names)
    train_data = MRIDataset(train_vol_names)
    num_train = len(train_vol_names)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = da.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=da.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=1)


    for epoch in range(args.epochs):

        # training
        loss = train(train_queue, model, optimizer, criterion, epoch, num_train)
        logging.info('train_loss of %03d epoch : %f', epoch, loss.item())

        save_file_name = os.path.join(args.save, '%d.ckpt' % epoch)
        torch.save(model.state_dict(), save_file_name)


def train(train_queue, model, optimizer, criterion, epoch, num_train):

    model.train()

    for step, (tgt, src) in enumerate(train_queue):  # atlas, X
        tgt = Variable(tgt, requires_grad=False).cuda()
        src = Variable(src, requires_grad=False).cuda()

        optimizer.zero_grad()
        y, flow, flow_pyramid = model.forward(tgt, src) 
        loss, ncc, grad = criterion(y, tgt, src, flow, flow_pyramid, hyper_1=10, hyper_2=15, hyper_3=3.2, hyper_4=0.8)

        loss.backward()
        optimizer.step()

        logging.info('train %03d epoch %03d step loss: %f, sim: %f, grad: %f', epoch, step, loss, ncc, grad)

        if step % args.report_freq == 0:
            i = (epoch + 1) * num_train + step
            writer.add_scalar('loss', loss, i)
            writer.add_scalar('sim_loss', ncc, i)
            writer.add_scalar('det_loss', grad, i)

    return loss


if __name__ == '__main__':
    main()
