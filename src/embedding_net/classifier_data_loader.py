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
    def __init__(self, utt_ids,voice_data,face_data,labels):
        self.face_data = face_data
        self.voice_data = voice_data
        self.utt_ids = utt_ids
        self.num_class = 5992
        

    def __len__(self):
        return len(self.face_data)

    def __getitem__(self, index):
        return (self.voice_data[self.utt_ids[index]],self.face_data[index],self.labels[index])

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
    
    train_utt_ids = []
    train_face_embeds=[]
    train_labels =[]
    valid_utt_ids =[]
    valid_face_embeds=[]
    valid_labels=[]
    train_spk_list = train_voice_list
    ## Get voice utt iD's , labels to extract feats from train_xvec
    for i in range(len(list(train_spk_list))):
    ## TRAIN
        utt_ids = trainval_spk2utt[train_spk_list[i]][:20]
        face_ids = face_embed_data[train_face_list[i]][:20]
        labels = [i] *20
        train_utt_ids.extend(utt_ids)
        train_face_embeds.extend(face_ids)
        train_labels.extend(labels)
    ## VALID
        utt_ids = trainval_spk2utt[train_spk_list[i]][20:25]
        if len(utt_ids) < 5:
        #print('INIT LEN ',len(utt_ids),utt_ids)
            diff = -(len(utt_ids)-5)
        #print('DIFF is',diff,"Factor is ",diff//len(utt_ids)+1)
            utt_ids = utt_ids * (diff//len(utt_ids)+1)
        #print('LATER LEN ',len(utt_ids),utt_ids)

            if len(utt_ids)-5 !=0:
                diff = -(len(utt_ids)-5)
            #print('SECOND DIFF is',diff)
                utt_ids = utt_ids + utt_ids[:diff]
            #print('FINAL LEN IS',len(utt_ids))
            assert(len(utt_ids)==5)
        #print(train_spk_list[i],len(trainval_spk2utt[train_spk_list[i]]))
        face_ids = face_embed_data[train_face_list[i]][20:25]
        labels = [i] *25
        valid_utt_ids.extend(utt_ids)
        valid_face_embeds.extend(face_ids)
        valid_labels.extend(labels)


    train_dataset = ClassifyDataset(train_utt_ids,train_xvec,train_face_embeds,train_labels)
    valid_dataset = ClassifyDataset(valid_utt_ids,train_xvec,valid_face_embeds,valid_labels)
    print('Creating loaders with batch_size as ', bs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=3,pin_memory= True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4,pin_memory = True)
    
    print("Time taken for data loading: ", time.time() - start)
    return train_loader,valid_loader,None
