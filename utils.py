from __future__ import print_function 
import os
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.models as models
import torch
import torch.nn as nn
import pickle
import pdb
import torch.optim as optim
from PIL import Image
import numpy as np
import random
import torch.backends.cudnn as cudnn
from time import time
from scipy.io import wavfile



def net_frozen(args, model):
    print('********************************************************')
    model.frozen_until(args.frozen_until)
    init_lr = args.lr
    if args.trainer.lower() == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                lr=init_lr, weight_decay=args.weight_decay)
    elif args.trainer.lower() == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                lr=init_lr,  weight_decay=args.weight_decay)
    print('********************************************************')
    return model, optimizer

def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return model

def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model

def second2str(second):
    h = int(second/3600.)
    second -= h*3600.
    m = int(second/60.)
    s = int(second - m*60)
    return "{:d}:{:02d}:{:02d} (s)".format(h, m, s)

def print_eta(t0, cur_iter, total_iter):
    """
    print estimated remaining time
    t0: beginning time
    cur_iter: current iteration
    total_iter: total iterations
    """
    time_so_far = time() - t0
    iter_done = cur_iter + 1
    iter_left = total_iter - cur_iter - 1
    second_left = time_so_far/float(iter_done) * iter_left
    s0 = 'Epoch: '+ str(cur_iter + 1) + '/' + str(total_iter) + ', time so far: ' \
        + second2str(time_so_far) + ', estimated time left: ' + second2str(second_left)
    print(s0)

def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
        else Variable(X)

def get_length_wav(fn):
    frame_rate, signal = wavfile.read(fn)
    return float(signal.shape[0])/frame_rate

