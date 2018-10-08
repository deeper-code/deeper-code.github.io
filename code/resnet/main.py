
# refrence :  https://github.com/pytorch/examples/blob/master/imagenet/main.py


import os 
import argparse
import shutil
import random 
import time

import torch
import torch.optim as optim
import torch.nn.parallel as parallel

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet


#parse = argparse()


if __name__ == '__main__':
    model = resnet.resnet18(pretrained=True, num_classes=2)
    