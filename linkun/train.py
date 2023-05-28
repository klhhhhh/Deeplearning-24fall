import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
import res2net


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='cifar100',
                        choices=['cifar100', 'imagenet'],
                        help='dataset name')
    parser.add_argument('--imagenet-dir', help='path to ImageNet directory')
    parser.add_argument('--arch', default='res2next29_6cx24wx6scale_se',
                        choices=res2net.__all__,
                        help='model architecture')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
    parser.add_argument('--milestones', default='150,225', type=str)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    args = parser.parse_args()

    return args