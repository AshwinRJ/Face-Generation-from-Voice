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

## LOAD SPEECH KALDI ARKS- Kaldi IO

""""
END OF UTILS FUNCTIONS
"""""

class ClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, face_data, face_list, voice_data, voice_list,spk2utt,mode="t"):
        self.face_data = face_data
        self.face_list = face_list
        self.voice_data = voice_data
        self.voice_list = voice_list
        self.spk2utt = spk2utt
        self.num_class = 5994
        self.mode = mode
        self.face_counts = np.zeros((self.num_class))
        self.voice_counts = np.zeros((self.num_class))
        if mode == 't':
            self.face_index_list = np.arange(30)
            self.voice_index_list = np.arange(20)
            self.num_face_repeats = 30 
            self.num_voice_repeats = 20
        else:
            self.face_counts = 30 * np.ones((self.num_class))
            self.num_face_repeats = 10
            self.num_voice_repeats = 10
        

    def __len__(self):
        return self.num_class * (self.num_face_repeats * self.num_voice_repeats)

    def __getitem__(self, index):
        index = int(index % self.num_class)
        face_id = self.face_list[index]
        voice_id = self.voice_list[index]
        if self.mode == "t":
            i = self.face_counts[index]
            self.face_counts+=1
            self.voice_counts+=1
        else:
            i = self.face_counts[index]
            j = self.voice_counts[index]
            if i > len(self.face_list[index]-31):
                i = np.random.randint(low=31,high=len(self.face_data[face_id]))
            if j > len(self.spk2utt[voice_id]) -1:
                j = np.random.randint(low=0,high=len(self.spk2utt[voice_id]))
            self.face_counts+=1
            self.voice_counts+=1

        face_embed = torch.Tensor(np.array(self.face_data[face_id][i]))
        voice_embed = torch.Tensor(np.array(self.voice_data[self.spk2utt[voice_id][j]]))
        assert(face_embed.size(0)==512 and voice_embed.size(0)==512)
        return (voice_embed, face_embed,index)
    
## Index- pair (f,c)
#Create list that maps face_vec_id,voice_vec_id,


def get_data_loaders(bs):
    start = time.time()
    # Load the data files
    common_meta = pd.read_csv('vox2_meta.csv')
    face_embed_data = load_json("../../../data/vggface2_voxceleb2_embeddings.json")

    # List of face and voice IDs
    # Contains the class names
    dont_include = ['n003729 ' , 'n003754 ']

    train_face_list, valid_face_list, test_face_list = [], [], []
    train_voice_list, valid_voice_list, test_voice_list = [], [], []

    for i in range(len(common_meta['Set '])):
        if common_meta['Set '].iloc[i] == "dev " and common_meta['VGGFace2 ID '].iloc[i] not in dont_include:
            train_face_list.append(common_meta['VGGFace2 ID '].iloc[i][:-1].strip())
            train_voice_list.append(common_meta['VoxCeleb2 ID '].iloc[i][:-1].strip())

        elif common_meta['Set '].iloc[i] == "test " and common_meta['VGGFace2 ID '].iloc[i][:-1] not in dont_include:
            test_face_list.append(common_meta['VGGFace2 ID '].iloc[i][:-1].strip())
            test_voice_list.append(common_meta['VoxCeleb2 ID '].iloc[i][:-1].strip())

    train_xvec = { key.strip():vec.tolist() for key,vec in read_vec_flt_ark('../../../data/xvec_v2_train.ark')}
    
    assert(len(list(train_xvec.keys()))==1092009)

    trainval_spk2utt = {line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open('../../../data/spk2utt_train','r').readlines()}
    assert(len(list(trainval_spk2utt.keys()))==5994)
    
    train_spk2utt = {spk:trainval_spk2utt[spk][:20] for spk in train_voice_list}
    valid_spk2utt = {spk:trainval_spk2utt[spk][20:] for spk in train_voice_list}
    
    ## For Test data


    train_dataset = ClassifyDataset(face_embed_data, train_face_list, train_xvec, train_voice_list,train_spk2utt,mode="t")
    valid_dataset = ClassifyDataset(face_embed_data, train_face_list, train_xvec, train_voice_list,valid_spk2utt,mode="v")
    print('Creating loaders with batch_size as ', bs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8,pin_memory= True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4,pin_memory = True)
    
    print("Time taken for data loading: ", time.time() - start)
    return train_loader,valid_loader,None
