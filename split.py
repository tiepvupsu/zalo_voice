"""
Split data and build data loader
"""
from __future__ import print_function 
import os
from torch.autograd import Variable
# from python_speech_features.base import mfcc 
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
# from whale_detector.helperFunctions import * 
import scipy
from scipy.io import wavfile
from scipy import signal
import librosa
# from models.research.audioset.vggish_input2 import waveform_to_examples
import config as cf
from collections import defaultdict
from torchvision import transforms
import h5py


def _fn_clustering(fns):
    fns.sort()
    groups = []
    for i, fn in enumerate(fns):
        fn_no_ext = fn.split('/')[-1].split('.')[0].strip('_').split('_') # avoid _ at the end
        prefix = '_'.join(fn_no_ext[:-1])
        num = fn_no_ext[-1]
        if i == 0 or len(num) > 3 or prefix != last: # new group 
            groups.append([fn])        
            last = prefix  
        else:
            groups[-1].append(fn)
    return groups


    
def _train_test_split_by(voice_fns, split_by = 'file', split_size = 0.1, offset = [0, 0]):
    """
    split by file or voice, return two list of filepaths and their nested ids


    INPUT:
        voice_fns: list of list of string
            [[fn0], [fn1, fn2]]: list of voice in voice_fns
        split_by: string
            split by 'file' (default) or 'voice'
        split_size: float or int 
            float, 0 < split_size < 1, portion of the first set 
            int, 1 < split_size < total_files, number of file in the first set
        offset: a list of two indexes
            offset of the first list and the second list 
    OUTPUT:
        fns1: list of filepaths
            [fn1_0, fn1_1]
        fns2: list of filepaths
            [fn2_0, fn2_1]
        nested_ids1: list of list of integer 
            nested ids, in the form [[id1_0, id1_1], [id1_2], ...]
        nested_ids2: list of list of integer
    """
    ## if split_by == 'voice'
    total_clusters = len(voice_fns)
    ids_mixed = np.random.permutation(total_clusters)
    if split_by=='voice':
        num_cluster1 = int(split_size*total_clusters) if 0 < split_size < 1 else split_size # split by file 
        # mix 
        set_fns = [[], []]
        set_nested_ids = [[], []]
        for i in ids_mixed:
            fns = voice_fns[i]
            idx = +(i > num_cluster1) # = 0 if i < num_cluster1
            set_fns[idx].extend(fns)
            set_nested_ids[idx].append([offset[idx] + j for j in range(len(fns))])
            offset[idx] += len(fns)
    else: # == 'files'
        total_files = sum(map(len, voice_fns))
        num_file1 = int(split_size*total_files) if 0 < split_size < 1 else split_size # split by file 
        set_fns = [[], []]
        set_nested_ids = [[], []]
        acc_sum = 0
        flag = False # in which set, when acc_sum >= num_file1, then 
        for i in ids_mixed:
            fns = voice_fns[i]
            idx = +(acc_sum > num_file1)
            acc_sum += len(fns)
            set_fns[idx].extend(fns)
            set_nested_ids[idx].append([offset[idx] + j for j in range(len(fns))])
            offset[idx] += len(fns)

    return set_fns[0], set_fns[1], set_nested_ids[0], set_nested_ids[1]


def shift_nested_ids(nested_ids, groups, offset = 0):
    """
    move nested_ids[group] for group in groups to start at offset
    INPUT:
        nested_ids: [[[0], [1, 2]], [[3, 4], [5, 6]], [7]]
        groups: [0, 2]
        offset: 0
    OUTPUT:
        res = [[[0], [1, 2]], [4]]
                               7
    """
    def helper(nested_id, offset):
        res = []
        for voice in nested_id:
            res.append([offset + i for i in range(len(voice))])
            offset += len(voice)
        return res, offset 
    res = []
    for group in groups:
        tmp, offset = helper(nested_ids[group], offset)
        res.append(tmp)
        # offset = res[-1][-1]+1
        # res.append(_res)
    return res 
        

def my_split(csv_file, select = 'all', split_by = 'file', split_to= 'train',\
     split_size = 0.9, random_state = None):
    """
    overall train split method for voice
    suppose that training data is store in the form 
    base_/
        cate1/
        cate2/
    Args:
        csv_file: string
            csv file that contains all filenames and labels (female_north, ...)
        select: string
            which classification problem we are considering.
            choices = ['gender', 'female', 'male', 'all'], default = 'all'
                'gender': female vs male (2 classes)
                'female': female_north, female_central and female_soutch (3 classes)
                'male': similar but with male (3 classes)
                'all': all (6 classes)
        cates: a list of strings
            subfolders cates = ['female_north', ...]
        split_by: string
            split by number of files or number of voice clusters
            choices = ['file', 'voice'], default 'file'
        split_to: string
            choices = ['train', 'test'], split by train or test set, default 'train'
        split_size: a number 
            if float, 0 < split_size < 1, split_size ~ portion of total as `split_to
            if int, 1 < split_size < min(num elements in each class), each class has    
            `split_size` elements in `split_to set.
        random_state: None or int 
            random seed
    Returns: 
        fns1: list of strings 
            list of all training filenames
        fns2: list of strings 
            list of all test filenamse
        lbs1: list of integers
            list of all training labels
        fns2: list of integers 
            list of all test labels
        nested_ids1, nested_ids2: list of lists of lists
            return a 6-element list, each element represents indexes one class
            Each element contain groups of ids from the same voice
            For example (for example, there are two classes): 
            nested_ids = [[[0], [1, 2], [3, 4, 5]], [[6], [7, 8, 9, 10]]]
            That means:
                [0, 1, 2, 3, 4, 5] from class 1 where [0] is voice 1, [1, 2] is voice 2,
                    [3, 4, 5] is voice 3 
                [6, 7, 8, 9, 10] from class 2 where [6] is a voice, [7, 8, 9, 10] from a voice
    """
    np.random.seed(seed = random_state)
    # fns[i] is a list of all filenames in class i  
    _fns = defaultdict(list)
    cnt  = 0 
    with open(csv_file) as fp:
        for line in fp:
            if cnt == 20270: break
            cnt += 1
            fn, lb = line.strip().split(',')
            # fn = os.path.join(cf.BASE_TRAIN, fn)
            # pdb.set_trace()
            fn = cf.BASE_TRAIN + fn 
            _fns[int(lb)].append(fn)
    fns = [] # fns[i] is a list of all filenames in class i  
    for cate in range(len(cf.CATES)):
        fns.append(_fns[cate]) 
    ## get all files
    fns1 = []
    fns2 = []
    lbs1 = []
    lbs2 = []
    nested_ids1 = []
    nested_ids2 = []
    offset = [0, 0]
    for i, fn_class in enumerate(fns):  
        _groups = _fn_clustering(fn_class)
        _fns1, _fns2, _nested_ids1, _nested_ids2 = \
            _train_test_split_by(_groups, split_by = split_by,\
            split_size = split_size, offset = offset)
        fns1.extend(_fns1)
        fns2.extend(_fns2)
        lbs1.extend([i]*len(_fns1))
        lbs2.extend([i]*len(_fns2))
        nested_ids1.append(_nested_ids1)
        nested_ids2.append(_nested_ids2)
        offset[0] = nested_ids1[-1][-1][-1] + 1
        offset[1] = nested_ids2[-1][-1][-1] + 1
    
    num_female1 = sum(1 for lb in lbs1 if lb < 3)
    num_female2 = sum(1 for lb in lbs2 if lb < 3)

    if select == 'gender':
        lbs1 = [lb//3 for lb in lbs1]
        lbs2 = [lb//3 for lb in lbs2]
        nested_ids1 = [nested_ids1[0] + nested_ids1[1] + nested_ids1[2],\
                       nested_ids1[3] + nested_ids1[4] + nested_ids1[5]]
        nested_ids2 = [nested_ids2[0] + nested_ids2[1] + nested_ids2[2],\
                       nested_ids2[3] + nested_ids2[4] + nested_ids2[5]]
    elif select == 'female':
        fns1 = fns1[: num_female1]
        fns2 = fns2[: num_female2]
        lbs1 = lbs1[: num_female1]
        lbs2 = lbs2[: num_female2]
        nested_ids1 = nested_ids1[:3]
        nested_ids2 = nested_ids2[:3]
    elif select == 'male':
        fns1 = fns1[num_female1:]
        fns2 = fns2[num_female2:]
        lbs1 = [lb%3 for lb in lbs1[num_female1:]]
        lbs2 = [lb%3 for lb in lbs2[num_female2:]]
        nested_ids1 = shift_nested_ids(nested_ids1, [3, 4, 5], offset = 0)
        nested_ids2 = shift_nested_ids(nested_ids2, [3, 4, 5], offset = 0)
    elif select == 'accent':
        raise NotImplementedError
    return fns1, fns2, lbs1, lbs2, nested_ids1, nested_ids2 
    

def load_fns_lbs_from_csv(csv_fn, merge = False, header = False):
    """
    csv_fn is path to csv file, no header
    """
    fns = []
    lbs = []
    # base_dir = cf.BASE_DIR_MERGED if merge else cf.Base
    header_flag = header
    with open(csv_fn) as fp:
        for line in fp:
            if header_flag:
                header_flag = False
                continue
            tmp = line.strip().split(',')
            file_path = tmp[0]
            label = int(tmp[1]) 
            fns.append(file_path)
            lbs.append(label)
    return fns, lbs
