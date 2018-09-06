from __future__ import division
from utils import *
from data import MyDatasetSTFT
from sklearn.model_selection import train_test_split
# from utils import *
import torch
import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
# import time
import torch.backends.cudnn as cudnn
import os, sys
from time import time, strftime

class MyNet(nn.Module):
    def __init__(self, num_classes=6):
        super(MyNet, self).__init__()
        n_filters = [1, 16, 32, 64, 128]
        # n_filters = [1, 32, 64, 128, 256]
        
        ks = [5, 5, 3, 3, 3] 
        pd = [k//2 for k in ks] 
        self.layers = []
        for i in range(1, len(n_filters)): 
            self.layers.append( nn.Conv2d(n_filters[i-1], n_filters[i], 
                kernel_size=ks[i-1], stride=1, padding=pd[i-1]))
            self.layers.append(nn.BatchNorm2d(n_filters[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv = nn.Sequential(*self.layers)
        self.fc = nn.Linear(7*12*n_filters[-1], num_classes)
        
    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(x.size(0), -1)
        out = self.fc(out)
        return out

class MyVGGNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained = True):
        super(MyVGGNet, self).__init__()
        if depth == 11:
            model = models.vgg11(pretrained)
        elif depth == 16:
            model = models.vgg16(pretrained)
        elif depth == 19:
            model = models.vgg19(pretrained)

        # pdb.set_trace()
        self.shared = model.features
        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, num_classes))
        # self.avgpool = nn.AvgPool2d(7)

        self.target = nn.Sequential(*feature_model)
        # feature_model.append(nn.Linear(cf.feature_size, len(dset_classes)))
        # model_ft.fc = nn.Sequential(*feature_model)

        # pdb.set_trace()
        # self.num_ftrs = model.fc.in_features
        # # self.num_classes = num_classes

        # self.shared = nn.Sequential(*list(model.children())[:-1])
        # self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()
        x = self.shared(x)
        x =  x.view(x.size(0), -1)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1
class MyResNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained = True):
        super(MyResNet, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)

        self.num_ftrs = model.fc.in_features
        # self.num_classes = num_classes

        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()

        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

class MyInceptionV3Net(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(MyInceptionV3Net, self).__init__()

        model = models.inception_v3(pretrained)
        pdb.set_trace()
        self.num_ftrs = model.fc.in_features
        # pdb.set_trace()
        # self.num_classes = num_classes
        self.shared = nn.Sequential(*list(model.children())[:-1])
        # self.extra = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size = (7, 7), padding = 0),
                # nn.ReLU(inplace = True))
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()
        x = self.shared(x)
        # x = self.extra(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

class MyDenseNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained = True):
        super(MyDenseNet, self).__init__()
        if depth == 121:
            model = models.densenet121(pretrained)
        elif depth == 161:
            model = models.densenet161(pretrained)
        elif depth == 169:
            model = models.densenet169(pretrained)
        elif depth == 201:
            model = models.densenet201(pretrained)

        self.num_ftrs = model.classifier.in_features
        # pdb.set_trace()
        # self.num_classes = num_classes
        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.extra = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size = (7, 7), padding = 0),
                nn.ReLU(inplace = True))
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()
        x = self.shared(x)
        x = self.extra(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)

        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each row of Z is a set of score.    
    """
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True)) 
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A


def predict(model, dset_loader, n_files, n_tests = 3, num_classes = 6):
    """
    Use a pretrained model to predict each input in dset 

    ----
    INPUT: 
        model: a pretrained model
        dset: a MyDatasetSTFT variable
    OUTPUT:
        pred_outputs: prediction results, sum of all ouput before softmax  
        pred_probs: prediction results, sum of all probability  
    """

    preds = np.zeros((n_files, n_tests))
    outputs = np.zeros((n_files, num_classes)) 
    probs = np.zeros((n_files, num_classes))
    fns = []

    for test in range(n_tests):
        tot = 0 
        print('test number: {}'.format(test))
        start = 0
        fni = []
        for batch_idx, (inputs, labels, fns0) in enumerate(dset_loader):
            # if batch_idx == 0:
            #     print(fns0[:10])
            n = len(fns0)
            inputs = cvt_to_gpu(inputs)
            output = model(inputs)
            _, pred  = torch.max(output.data, 1)
            preds[start:start + n, test] = pred.data.cpu().numpy()
            outputs[start:start + n, :] += output.data.cpu().numpy()
            start += n
            tot += len(fns0)
            fni.extend(fns0)
        fns.append(fni)
            # print('processed {}/{}'.format(tot, n_files))

        probs += softmax_stable(outputs)

    pred_output = np.argmax(outputs, axis = 1)
    pred_probs = np.argmax(probs, axis = 1)
    return pred_output, pred_probs, fns

