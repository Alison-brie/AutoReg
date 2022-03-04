# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************


import os
import sys
import utils
import glob
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from model import *
import dataset
import random
import torch.utils.data as da
import genotypes
from losses import LossFunction_mpr
from torch.autograd import Variable
from visdom import Visdom

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')


parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='models/2-train-Modified', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.95, help='portion of training data')

parser.add_argument('--arch', type=str, default='Search_ABIDE', help='which architecture to use')
parser.add_argument('--gamma', type=float, default=0, help='learning rate decay')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--shape', type=list, default=[160, 192, 224], help='weight decay for arch encoding')
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
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    criterion = LossFunction_mpr(args.shape).cuda()

    model = Network(criterion, 'Concat_Cell', genotype, args.shape)
    model = model.cuda()
    # model.load_state_dict(torch.load('models/2-train-Modified/157.ckpt'))

    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     model = nn.DataParallel(model, device_ids=[0, 3])
    #     model.to(device)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # data generator
    data_dir = 'E:/Lzy/data/ABIDE/train'
    train_vol_names = glob.glob(os.path.join(data_dir, '*.nii.gz'))
    random.shuffle(train_vol_names)
    train_data = datagenerators.MRIDataset(train_vol_names, atlas_file='E:/Lzy/NAS-0512/data/atlas_norm.npz')

    num_train = len(train_vol_names)
    indices = list(range(num_train))
    train_portion = 0.9
    split = int(np.floor(train_portion * num_train))

    train_queue = da.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=da.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=1)

    # Loss plot
    # viz = Visdom()
    # 创建窗口并初始化
    # viz.line([0.], [0], win='baseline_NAS', opts=dict(title='baseline_NAS'))

    for epoch in range(args.epochs):
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        loss, ncc, mind, grad = train(train_queue, model, optimizer, epoch)
        logging.info('train epoch-%03d - average- ,  %f, %f, %f, %f', epoch, loss, ncc, mind, grad)

        tmp = epoch
        save_file_name = os.path.join(args.save, '%d.ckpt' % tmp)
        torch.save(model.state_dict(), save_file_name)

        # python -m visdom.server
        # viz.line([loss.item()], [epoch], win='baseline_NAS', update='append')


def train(train_queue, model, optimizer, epoch):
    train_loss, train_ncc, train_mind, train_grad = [], [], [], []
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda()
        input = input.cuda()
        input = Variable(input)
        target = Variable(target)

        optimizer.zero_grad()
        loss, ncc, mind, grad = model._loss(input, target)

        train_loss.append(loss.item())
        train_ncc.append(ncc.item())
        train_mind.append(mind.item())
        train_grad.append(grad.item())

        loss.backward()
        optimizer.step()

        if step % args.report_freq == 0:
            logging.info('train epoch-%03d ,step-%03d  %f, %f, %f, %f', epoch, step, loss, mind, ncc, grad)

    return np.average(train_loss), np.average(train_ncc), np.average(train_mind), np.average(train_grad)


if __name__ == '__main__':
    main()
