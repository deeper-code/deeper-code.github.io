
# refrence :  https://github.com/pytorch/examples/blob/master/imagenet/main.py


import os 
import argparse
import shutil
import random 
import time
from PIL import Image
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel as parallel
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import resnet


import matplotlib.pyplot as plt
import collections
import math

from utils import Measure

parse = argparse.ArgumentParser()

parse.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parse.add_argument('--print_freq', default=10, type=int, help='print frequency (defalut: 10).')
parse.add_argument('--seed', default=None, type=int, help='random seed.')
parse.add_argument('--lr', default=1e-4, type=float, help='learning rate.')
parse.add_argument('--batch_size', default=32, type=int, help='batch size.')
parse.add_argument('--resume', default=None, type=str, help='resume from a checkpoint to continue training.')
parse.add_argument('--train_data', default=None, type=str, help='the path of train data.')
parse.add_argument('--valid_rate', default=0.1, type=float, help='rate of validation/training')
parse.add_argument('--test_data', default=None, type=str, help='the path of test data.')
parse.add_argument('--action', default='train', type=str, help='do what? [train, valid, test]')
parse.add_argument('--epochs', default=3, type=int, help='totle epoch for training.')


args = parse.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "1"






def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



class myDataset(Dataset):
    def __init__(self, file_list, trans):
        self.file_list = file_list
        self.trans = trans 

    def __getitem__(self, index):
        filename = self.file_list[index]
        clazz = os.path.basename(filename).split('.')[0]
        return self.trans(Image.open(filename)), ['dog', 'cat'].index(clazz)

    def __len__(self):
        return len(self.file_list)



def train(data_loader, model, criterion, optimizer, epoch):
    batch_time = Measure('train@batch_time', is_plot=False)
    data_time  = Measure('train@data_time', is_plot=False)
    losses     = Measure('train@loss')
    top1       = Measure('train@top1-accuray')
    #top5       = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time 
        data_time.update(time.time() - end)

        if args.gpu is not None :
            input  = input.cuda(args.gpu, non_blocking=True) # non_blocking: 不分块 
            target = target.cuda(args.gpu, non_blocking=True)

        # Computes output and loss
        output = model(input)
        loss   = criterion(output, target) # output(onehot), target(int)

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))

        # update average meter.
        losses.update(loss.item(), n=input.size(0))
        top1.update(prec1[0].cpu().numpy()[0], n=input.size(0))
        #top5.update(prec5[0], n=input.size(0))

        # Computes gradient and do `optimizer` step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



        if i % args.print_freq == 0:
            #print(type(batch_time), type(data_time), type(losses), type(top1))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(data_loader), batch_time=batch_time, 
                   data_time = data_time, loss=losses, top1= top1))




def validate(data_loader, model, criterion, epoch):
    batch_time = Measure('val@batch_time', is_plot=False)
    data_time  = Measure('val@data_time', is_plot=False)
    losses     = Measure('val@loss')
    top1       = Measure('val@top1-accuray')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            data_time.update(time.time() - end)

            # compute output and loss
            output = model(input)
            loss   = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), n=input.size(0))
            top1.update(prec1[0].cpu().numpy()[0], n=input.size(0))
            #top5.update(prec5[0], n=input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(data_loader), batch_time=batch_time, 
                       data_time = data_time, loss=losses, top1= top1))           
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, is_lowest,  filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace(
                                os.path.basename(filename), 
                                'model_best.pth.tar'))





def main():

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    model = resnet.resnet50(pretrained=False, num_classes=2)

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = optim.SGD(model.parameters(), args.lr, 
                          momentum=0.9, weight_decay=1e-5)

    lowest_loss = 9999999.0
    best_prec1  = 0.0
    args.start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print('==> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1  = checkpoint['best_prec1']
            lowest_loss = checkpoint['lowest_loss']
            lr = checkpoint['lr']
            # load model weights
            model.load_state_dict(checkpoint['state_dict'])
            # load optimizer weights
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('==> loaded checkpoint {} (epoch {})'.format(
                   args.resume, args.start_epoch))
        else:
            print('==> no checkpoint found at "{}"'.format(args.resume))




    # Data loader
    all_files = list(glob.glob(args.train_data + '/*.jpg'))
    valid_nr  = int(len(all_files)*args.valid_rate)
    np.random.shuffle(all_files)

    train_files = all_files[valid_nr:]
    valid_files = all_files[:valid_nr]


    train_dataset = myDataset(train_files, 
                              transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.41695606, 0.45508163, 0.48832284],
                                                     std  = [0.25891816, 0.25640547, 0.26287844],)
                            ]))

    valid_dataset = myDataset(valid_files, 
                              transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.41695606, 0.45508163, 0.48832284],
                                                     std  = [0.25891816, 0.25640547, 0.26287844],)
                            ]))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=True, sampler=None)

    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)

    # do action
    if args.action == 'train':
        for epoch in range(args.start_epoch, args.epochs):
            # adjust learning rate
            #lr = adjust_learning_rate(optimizer, epoch)

            # training one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation-set
            acc, loss = validate(valid_loader, model, criterion, epoch)

            # save model if needed.
            is_best   = best_prec1 < acc 
            is_lowest = lowest_loss > loss 

            best_prec1 = max(best_prec1, acc)
            lowest_loss = min(lowest_loss, loss)

            checkpoint = {
                'epoch' : epoch,
                'best_prec1': best_prec1,
                'lowest_loss' : lowest_loss,
                'lr' : 0,
                'state_dict' : model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, is_best, is_lowest, filename='../checkpoints/checkpoint.pth.tar')
            Measure.summary()
            Measure.plot()


    else :
        print('==> validation and test mode do not support now.')
        return




if __name__ == '__main__':
    # pytorch 迁移学习是先加载1000分类的模型
    # model = resnet.resnet18(pretrained=True)
    # 然后修改最后的全连接层即可
    # model.fc = nn.Linear(512, 2)
    main()












