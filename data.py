from __future__ import print_function 
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from time import time
from collections import Counter 
from scipy.io import wavfile
from scipy import signal
import scipy
import config as cf
from split import my_split

def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann', # 'text'
                                        nperseg=nperseg,
                                        noverlap = noverlap,
                                        detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

def random_segment(audio_signal, N):
    length = audio_signal.shape[0]
    if N < length:
        start = random.randint(0, length - N)
        audio_signal = audio_signal[start:start + N]
    else: 
        tmp = np.zeros((N,))
        start = random.randint(0, N - length)
        tmp[start: start + length] = audio_signal 
        audio_signal = tmp
        # test_sound = np.pad(test_sound, (N - test_sound.shape[0])//2, mode = 'constant')
    return audio_signal


def gen_spec(wav_path, duration):
    samplerate, test_sound = wavfile.read(wav_path)
    test_sound =test_sound
    N = int(duration*samplerate)
    segment_sound = random_segment(test_sound, N)
    spectrogram = log_specgram(segment_sound, samplerate).astype(np.float32)

    # convert to ResNet input
    spectrogram = scipy.misc.imresize(spectrogram, (224, 224), interp='bicubic')
    out = np.zeros((3, 224, 224), dtype = np.float32)
    out[0, :, :] = spectrogram
    out[1, :, :] = spectrogram 
    out[2, :, :] = spectrogram 
    return out 
    
class MyDatasetSTFT(Dataset):
    def __init__(self, filenames, labels, transform=None, duration=2, test=False):
        assert len(filenames) == len(labels), "Number of files != number of labels"
        self.fns = filenames
        self.lbs = labels
        self.transform = transform
        self.duration = duration  # audio duration in second
        self.test = test
        self.root_test = "./data/wav" + cf.RATE + "/test/"

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        if self.test:
            fname = self.fns[idx]
            # fname = self.fns[idx].split("/")[-1]
            # fname = self.root_test + fname
        else:
            fname = self.fns[idx]
            if not os.path.isfile(fname):
                fname = self.fns[idx].split("/")[-1]
                fname = self.root_test + fname
            
        feats = gen_spec(fname, self.duration)
        return feats, self.lbs[idx], self.fns[idx]


def build_dataloaders(args):
    fns = []
    lbs = []
    # train
    with open(cf.TRAINING_GT) as fp:
        for line in fp:
            fn, lb = line.strip().split(',')
            fns.append(cf.BASE_TRAIN + fn)
            lbs.append(int(lb))
    train_fns, val_fns, train_lbs, val_lbs, _, _ = \
            my_split(cf.TRAINING_GT,\
                    split_size = 1 - args.val_ratio,\
                    random_state = args.random_state)
    # public test 
    semi_df = pd.read_csv(cf.TEST_GT)
    semi_df["id"] = semi_df["id"] + ".wav"
    semi_df["label"] = semi_df["gender"] * 3 + semi_df["accent"]
    semi_fns = semi_df["id"].values
    semi_lbs = semi_df["label"].values
    n_semi_data = len(semi_fns)
    n_semi_use = 1.0 * n_semi_data
    semi_use_idx = []
    while len(semi_use_idx) < n_semi_use:
        random_idx = random.randint(0, n_semi_data-1)
        if not random_idx in semi_use_idx:
            semi_use_idx.append(random_idx)
    use_semi_fns = list(semi_fns[semi_use_idx])
    use_semi_lbs = list(semi_lbs[semi_use_idx])
    semi_fns = [cf.BASE_PUBLIC_TEST + fn for fn in semi_fns] 
    use_semi_fns = [cf.BASE_PUBLIC_TEST + fn for fn in use_semi_fns]

    print('First val fn: {}'.format(val_fns[0]))
    train_fns += use_semi_fns
    train_lbs += use_semi_lbs
    print(Counter(train_lbs))
    print(Counter(val_lbs))

    num_classes = len(set(train_lbs))
    print('Total training files: {}'.format(len(train_fns)))
    print('Total validation files: {}'.format(len(val_fns)))
    print('Total classes: {}'.format(num_classes))

    dsets = dict()
    dsets['train'] =  MyDatasetSTFT(train_fns, train_lbs, duration = args.duration)
    dsets['val'] =  MyDatasetSTFT(val_fns, val_lbs, duration = args.duration)
    dsets['test'] = MyDatasetSTFT(semi_fns, semi_lbs, duration = args.duration)
    
    dset_loaders = dict() 
    dset_loaders['train'] = DataLoader(dsets['train'],
            batch_size = args.batch_size,
            shuffle = True,
            # sampler = WeightedRandomSampler(durations, args.batch_size, replacement = False),
            # sampler = StratifiedSampler_weighted_duration(train_fns, gamma = args.gamma),
            # sampler = StratifiedSampler_weighted(train_lbs, batch_size = args.batch_size, gamma = args.gamma),
            num_workers = cf.NUM_WORKERS)

    dset_loaders['val'] = DataLoader(dsets['val'],
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = cf.NUM_WORKERS)

    dset_loaders['test'] = DataLoader(dsets['test'],
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = cf.NUM_WORKERS)
    return dset_loaders, (train_fns, semi_fns, val_fns, train_lbs, semi_lbs, val_lbs)