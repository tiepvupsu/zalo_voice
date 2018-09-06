from __future__ import division
from sklearn.model_selection import train_test_split
# import dataloader
from data import MyDatasetSTFT
import torch
import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
import torch.backends.cudnn as cudnn
import os, sys
import config as cf
from time import time, strftime
from nets import MyNet, predict, softmax_stable
import predicts
from utils import *
import pandas as pd

parser = argparse.ArgumentParser(description='Zalo Voice Gender')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=18, choices = [18, 50, 152], type=int, help='depth of model')
parser.add_argument('--model_path', type=str, default = ' ')
parser.add_argument('--batch_size', type=int, default = 63)
parser.add_argument('--num_tests', type=int, default = 10, help = 'number of tested windows in each file')
args = parser.parse_args()

################# model preparation ########################
model_path_fns = ['./saved_model/model_' + str(i) + '.t7' for i in range(5)]


############ load models ###################### 
models = []
clippeds = [] ## each model might received a different input lengh
for model_path_fn in model_path_fns:
    # print("| Load pretrained at  %s..." % model_path_fn)
    checkpoint = torch.load(model_path_fn, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model = unparallelize_model(model)
    model = parallelize_model(model)
    best_acc = checkpoint['acc']
    clipped = checkpoint['args'].duration
    print('model {}, acc on CV: {}'.format(model_path_fn, best_acc))
    # print('previous acc\t%.4f'% best_acc)
    models.append(model)
    clippeds.append(clipped)

#############################################################
############# build dset_loaders  
print('Data preparation')
fns = [os.path.join(cf.BASE_PRIVATE_TEST, fn) 
        for fn in os.listdir(cf.BASE_PRIVATE_TEST)]
print('Total provided files: {}'.format(len(fns)))
lbs = [-1]*len(fns) # random labels, we won't use this
dset_loaders = []
for clipped in clippeds:
    dset = MyDatasetSTFT(fns, lbs, duration = clipped, test=True)
    dset_loaders.append(torch.utils.data.DataLoader(dset, 
                        batch_size=args.batch_size,
                        shuffle= False, num_workers=cf.NUM_WORKERS))

############# make predictions based on models and loaders  
pred_score, pred_probs, fns = \
        predicts.multimodels_multiloaders_class(models, dset_loaders, num_tests = args.num_tests)
#############################################################

############# write to csv file 
def gen_outputline(fn, pred):
    s = fn.split('/')[-1][:-4] + ','+ str(pred//3) + ',' +  str(pred%3) + '\n'
    print(s)
    return s 

# submission_fn = './result/submission_' + strftime('%m%d_%H%M') + '.csv'
submission_fn = './result/submission.csv'
f = open(submission_fn, 'w')
header = 'id,gender,accent\n'
f.write(header)
for i, fn in enumerate(fns):
    f.write(gen_outputline(fn, pred_score[i]))

f.close()
print('done')

 
