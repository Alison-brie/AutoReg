# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************

import os
import sys
base_dir = (os.path.dirname(os.path.dirname(os.path.abspath((__file__)))))
sys.path.append(base_dir)

import random
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data as da
from torch.optim import Adam
from torch.autograd import Variable
from upperoptimizer import HyperGradient
from losses import LossFunction_ncc, LossFunction_dice, LossFunction_mind
from dataset import *
from model import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=base_dir +'/data/ADNI_test_data_with_mask.npz',
                    help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=45, help='num of training epochs')
parser.add_argument('--save', type=str, default='MPRNet-ST-img2atlas-ADNI-v5', help='experiment name')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-3, help='learning rate for upper') # 1e-3  5e-3 4e-3
parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for upper')
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

    criterion_reg = LossFunction_ncc().cuda()
    # criterion_reg = LossFunction_mind().cuda()
    # criterion_reg = LossFunction_mixed().cuda()
    criterion_seg = LossFunction_dice().cuda()

    model = MPRNet_ST(criterion_reg, criterion_seg).cuda()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # data generator
    train_vol_names = np.load(args.data_dir)['arr_0']
    random.shuffle(train_vol_names)
    train_data = MRIDatasetWithMask(train_vol_names)
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

    hyper_optimizer = HyperGradient(model, args)

    for epoch in range(args.epochs):
        lr = args.learning_rate

        # training
        train(train_queue, valid_queue, model, hyper_optimizer, optimizer, lr, epoch, num_train)

        logging.info('hyper1 of %03d epoch : %f', epoch, model.hyper_1.item())
        logging.info('hyper2 of %03d epoch : %f', epoch, model.hyper_2.item())
        logging.info('hyper3 of %03d epoch : %f', epoch, model.hyper_3.item())
        logging.info('hyper4 of %03d epoch : %f', epoch, model.hyper_4.item())
        # logging.info('hyper5 of %03d epoch : %f', epoch, model.hyper_5.item())

        writer.add_scalar('hyper_1', model.hyper_1.item(), epoch)
        writer.add_scalar('hyper_2', model.hyper_2.item(), epoch)
        writer.add_scalar('hyper_3', model.hyper_3.item(), epoch)
        writer.add_scalar('hyper_4', model.hyper_4.item(), epoch)
        # writer.add_scalar('hyper_5', model.hyper_5.item(), epoch)

        save_file_name = os.path.join(args.save, '%d.ckpt' % epoch)
        torch.save(model.state_dict(), save_file_name)


def train(train_queue, valid_queue, model, hyper_optimizer, optimizer, lr, epoch, num_train):
    for step, (target, source, _, _) in enumerate(train_queue):
        model.train()

        target = Variable(target, requires_grad=False).cuda()
        source = Variable(source, requires_grad=False).cuda()

        # get a random minibatch from the search queue with replacement
        target_search, source_search, target_mask_search, source_mask_search = next(iter(valid_queue))
        target_search = Variable(target_search, requires_grad=False).cuda()
        source_search = Variable(source_search, requires_grad=False).cuda()
        target_mask_search = Variable(target_mask_search, requires_grad=False).cuda()
        source_mask_search = Variable(source_mask_search, requires_grad=False).cuda()

        if epoch>=15:
            hyper_optimizer.step(target, source, target_search, source_search, target_mask_search, source_mask_search,
                             lr, optimizer, unrolled=args.unrolled)

        # hyper_optimizer.step(target, source, target_search, source_search, target_mask_search, source_mask_search,
        #                      lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        loss, ncc, grad = model._lower_loss(target, source)
        upper_loss = model._upper_loss(target_search, source_search, target_mask_search, source_mask_search)

        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        
        logging.info('train %03d epoch %03d step loss: %f, ncc: %f, grad: %f,  upper_loss: %f', 
                               epoch, step, loss, ncc, grad, upper_loss)
        

        if step % args.report_freq == 0:
            i = (epoch + 1) * num_train + step
            writer.add_scalar('loss', loss, i)
            writer.add_scalar('sim_loss', ncc, i)
            writer.add_scalar('det_loss', grad, i)
            writer.add_scalar('upper_loss', upper_loss, i)
    return

if __name__ == '__main__':
    main()
