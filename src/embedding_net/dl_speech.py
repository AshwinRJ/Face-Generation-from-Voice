import os
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from kaldi_io import read_vec_flt,read_vec_flt_ark
import json
import logging
import itertools
torch.backends.cudnn.benchmark = True
from random import shuffle
import random
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""========================================================================="""
# Helper Functions


def load_json(filename='data.json'):
    with open(filename, 'r') as outfile:
        data = json.load(outfile)
    return data


def write_to_json(data, filename='data.json'):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

""""
===============================================================================================
"""""
class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, voice_data, voice_list,voice_labels):
        self.voice_data = voice_data
        self.voice_list = voice_list
        self.voice_labels = voice_labels
        self.num_class = 5992 #len(voice_list)

    def __len__(self):
        return len(self.voice_list) #self.num_class * (self.num_face_repeats * self.num_voice_repeats)


    def __getitem__(self, index):
        voice_feat = np.array(self.voice_data[self.voice_list[index]])
        assert(voice_feat.shape[0]==512)
        assert(len(voice_feat)==512)
        voice_label = self.voice_labels[index]
        assert(voice_label <= 5992)
        #print(voice_feat.shape,"VOICE",voice_label)
        return torch.from_numpy(voice_feat).float(),torch.tensor(voice_label).long()


def get_data_loaders(bs):
    start = time.time()
    # Load the data files
    common_meta = pd.read_csv('vox2_meta.csv')

    # List of face and voice IDs
    # Contains the class names
    dont_include = ['n003729 ' , 'n003754 ']

    train_face_list, valid_face_list, test_face_list = [], [], []
    train_voice_list, valid_voice_list, test_voice_list = [], [], []

    for i in range(len(common_meta['Set '])):
        if common_meta['Set '].iloc[i] == "dev " and common_meta['VGGFace2 ID '].iloc[i] not in dont_include:
            train_voice_list.append(common_meta['VoxCeleb2 ID '].iloc[i][:-1].strip())

        elif common_meta['Set '].iloc[i] == "test " and common_meta['VGGFace2 ID '].iloc[i][:-1] not in dont_include:
            test_voice_list.append(common_meta['VoxCeleb2 ID '].iloc[i][:-1].strip())

    train_xvec = { key.strip():vec.tolist() for key,vec in read_vec_flt_ark('../../../data/xvec_v2_train.ark')}
    assert(len(list(train_xvec.keys()))==1092009)

    trainval_spk2utt = {line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open('../../../data/spk2utt_train','r').readlines()}
    assert(len(list(trainval_spk2utt.keys()))==5994)
    voice_utts =[]
    voice_labels=[]
    for e,i in enumerate(train_voice_list):
        voice_utts.extend(trainval_spk2utt[i])
        voice_labels.extend([int(e)]*len(trainval_spk2utt[i]))
    print(len(voice_utts),len(voice_labels),"LABELS AND LIST")
    
    total_data = [(voice_utts[i],voice_labels[i]) for i in range(len(voice_utts))]
    random.shuffle(total_data)
    print(len(total_data))
    train_full = total_data[:-20000]
    valid_full = total_data[-20000:]
    print('LEN TRAIN',len(train_full),'LEN VALID',len(valid_full))
    train_utts = [data[0] for data in train_full]
    train_labels = [data[1] for data in train_full]
    dev_utts = [data[0] for data in valid_full]
    dev_labels = [data[1] for data in valid_full]

    train_dataset = EmbedDataset(train_xvec, train_utts,train_labels)
    valid_dataset = EmbedDataset(train_xvec, dev_utts,dev_labels)
    print('Creating loaders with batch_size as ', bs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory= True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory = True)
    print("Time taken for data loading: ", time.time()-start)
    return train_loader,valid_loader,None

