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

"""
Predict output of a new sample given model/models and fn/fns/dset
"""

def softmax_stable(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each row of Z is a set of score.    
    """
    e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True)) 
    A = e_Z / e_Z.sum(axis = 1, keepdims = True)
    return A

def loader_len(dset):
    """
    return len of a DataLoader 

    dset: a DataLoader that return (data, lbs, fns) at a batch
    """
    res = 0 
    for _, _, fns in dset:
        res += len(fns)
    return res 

def get_num_classes(model):
    """
    return number of output class of a pytorch classification model
    """
    # Step 1: unparalell_model
    model = unparallelize_model(model)
    return model.target[0].out_features


def singlemodel_score(model, dset_loader, num_tests = 1):
    """
    Use ONE pretrained model to predict score and probs each input in dset 
    Make multiples predictions and accumulate results

    ----
    INPUT: 
        model: a pretrained model
        dset: a MyDatasetSTFT variable
    OUTPUT:
        pred_outputs: np array -- prediction results, sum of all ouput before softmax  
        pred_probs: np array -- prediction results, sum of all probability  
        fns: list of filenames in loader order
    """
    num_files = loader_len(dset_loader)
    # num_classes = get_num_classes(model)
    num_classes = 6 

    # preds = np.zeros((num_files, n_tests))
    total_scores = np.zeros((num_files, num_classes)) 
    total_probs = np.zeros((num_files, num_classes))
    fns = []
    torch.set_grad_enabled(False)
    model.eval()

    for test in range(num_tests):
        tot = 0 
        print('test {}/{}'.format(test + 1, num_tests))
        start = 0
        for batch_idx, (inputs, labels, fns0) in enumerate(dset_loader):
            n = len(fns0)
            inputs = cvt_to_gpu(inputs)
            output = model(inputs)
            output = output.view((-1, num_classes))
            _, pred  = torch.max(output.data, 1)
            # preds[start:start + n, test] = pred.data.cpu().numpy()
            # pdb.set_trace()
            total_scores[start:start + n, :] += output.data.cpu().numpy()
            start += n
            tot += len(fns0)
            if test == 0: fns.extend(fns0)

        total_probs += softmax_stable(total_scores)
    return total_scores, total_probs, fns


def singlemodel_class(model, dset_loader, num_tests = 1):
    """
    predict label for dset_loader using one model
    This one is done after calling sm_score(model, dset_loader, num_tests)
    and get the torch.max(out.data, 1)
    """
    total_scores, total_probs, fns = singlemodel_score(model, dset_loader, num_tests)
    score_class = np.argmax(total_scores, axis = 1) 
    prob_class = np.argmax(total_probs, axis = 1)
    return score_class, prob_class, fns

def multimodels_score(models, dset_loader, num_tests = 1):
    """
    accumulate results from multiple models 
    output total scores and probabilities
    ----------
    INPUT:
        model: a list of pytorch classification models 
    """
    num_models = len(models)
    num_files = loader_len(dset_loader)
    num_classes = get_num_classes(model)
    # preds = np.zeros((num_files, n_tests))
    total_scores = np.zeros((num_files, num_classes)) 
    total_probs = np.zeros((num_files, num_classes))
    for model in models:
        tmp_score, tmp_prob, fns = singlemodel_score(model, dset_loader, num_tests)
        total_scores += tmp_score 
        total_probs += tmp_prob

    return total_scores, total_probs, fns


def multimodels_class(models, dset_loader, num_tests = 1):
    total_scores, total_probs, fns = multimodels_score(models, dset_loader, num_tests)
    score_class = np.argmax(total_scores, axis = 1) 
    prob_class = np.argmax(total_probs, axis = 1)
    return score_class, prob_class, fns

def multimodels_multiloaders_score(models, dset_loaders, num_tests = 1):
    """
    accumulate results from multiple models, each model uses one dsetloader
    number of models == number dset_loaders
    output total scores and probabilities
    ----------
    INPUT:
        model: a list of pytorch classification models 
    """
    assert len(models) == len(dset_loaders)
    num_models = len(models)
    num_files = loader_len(dset_loaders[0])
    num_classes = get_num_classes(models[0])
    # preds = np.zeros((num_files, n_tests))
    total_scores = np.zeros((num_files, num_classes)) 
    total_probs = np.zeros((num_files, num_classes))
    cnt = 0
    for model, dset_loader in zip(models, dset_loaders):
        cnt += 1
        print('######################')
        print('Model {}/{}'.format(cnt, len(models)))
        tmp_score, tmp_prob, fns = singlemodel_score(model, dset_loader, num_tests)
        # pdb.set_trace()
        total_scores += tmp_score 
        total_probs += tmp_prob

    return total_scores, total_probs, fns

def multimodels_multiloaders_class(models, dset_loaders, num_tests = 1):
    total_scores, total_probs, fns =\
        multimodels_multiloaders_score(models, dset_loaders, num_tests)
    score_class = np.argmax(total_scores, axis = 1) 
    prob_class = np.argmax(total_probs, axis = 1)
    return score_class, prob_class, fns


if __name__ == '__main__':
    model_fn = './checkpoint/resnet_18_0812_1250.t7'
    checkpoint = torch.load(model_fn, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    print(get_num_classes(model))
