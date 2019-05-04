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
import struct
import sys
import re
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
def open_or_fd(file, mode='rb'):
    """ fd = open_or_fd(file)
     Open file, gzipped file, pipe, or forward the file-descriptor.
     Eventually seeks in the 'file' argument contains ':offset' suffix.
    """
    offset = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
            (prefix,file) = file.split(':',1)
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file,offset) = file.rsplit(':',1)
        # input pipe?
        if file[-1] == '|':
            fd = popen(file[:-1], 'rb') # custom,
        # output pipe?
        elif file[0] == '|':
            fd = popen(file[1:], 'wb') # custom,
        # is it gzipped?
        elif file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset != None: fd.seek(int(offset))
    return fd


def read_key(fd):
    """ [key] = read_key(fd)
     Read the utterance-key from the opened ark/stream descriptor 'fd'.
    """
    assert('b' in fd.mode), "Error: 'fd' was opened in text mode (in python3 use sys.stdin.buffer)"

    key = ''
    while 1:
        char = fd.read(1).decode("latin1")
        if char == '' : break
        if char == ' ' : break
        key += char
    key = key.strip()
    if key == '': return None # end of file,
    assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
    return key


def read_vec_flt_ark(file_or_fd):
    """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
     Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.
     Read ark to a 'dictionary':
     d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_flt(fd)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def read_vec_flt(file_or_fd):
    """ [flt-vec] = read_vec_flt(file_or_fd)
     Read kaldi float vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode()
    if binary == '\0B': # binary flag
        ans = _read_vec_flt_binary(fd)
    else:    # ascii,
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove('['); arr.remove(']') # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=float)
    if fd is not file_or_fd : fd.close() # cleanup
    return ans

def _read_vec_flt_binary(fd):
    header = fd.read(3).decode()
    if header == 'FV ' : sample_size = 4 # floats
    elif header == 'DV ' : sample_size = 8 # doubles
    else : raise UnknownVectorHeader("The header contained '%s'" % header)
    assert (sample_size > 0)
    # Dimension,
    assert (fd.read(1).decode() == '\4'); # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
    if vec_size == 0:
        return np.array([], dtype='float32')
    # Read whole vector,
    buf = fd.read(vec_size * sample_size)
    if sample_size == 4 : ans = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8 : ans = np.frombuffer(buf, dtype='float64')
    else : raise BadSampleSize
    return ans

""""
END OF UTILS FUNCTIONS
"""""

class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, face_data, face_list, voice_data, voice_list,spk2utt):
        self.face_data = face_data
        self.face_list = face_list
        self.voice_data = voice_data
        self.voice_list = voice_list
        self.spk2utt = spk2utt
        self.num_class = len(voice_list)
        self.num_face_repeats = 30 ## Number of times I go through examples 
        self.num_voice_repeats = 30

    def __len__(self):
        return self.num_class * (self.num_face_repeats * self.num_voice_repeats)

    def __getitem__(self, index):
        index = int(index % self.num_class)
        face_id = self.face_list[index]
        voice_id = self.voice_list[index]

        i = np.random.randint(low=0,high=min(self.num_face_repeats,len(self.face_data[face_id])))
        j = np.random.randint(low=0,high=min(self.num_voice_repeats,len(self.spk2utt[voice_id])))

        face_embed = torch.Tensor(np.array(self.face_data[face_id][i]))
        voice_embed = torch.Tensor(np.array(self.voice_data[self.spk2utt[voice_id][j]]))
        assert(face_embed.size()[0]==512 and voice_embed.size()[0]==512)
            
        return (voice_embed, face_embed)
    
## Index- pair (f,c)
#Create list that maps face_vec_id,voice_vec_id,

def custom_collate(batch):
    voice_batch_embeddings = [item[0] for item in batch]
    face_batch_embeddings = [item[1] for item in batch]
    
    return voice_batch_embeddings, face_batch_embeddings

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

    valid_face_list = train_face_list[-200:]
    train_face_list = train_face_list[:-200]
    valid_voice_list = train_voice_list[-200:]
    train_voice_list = train_voice_list[:-200]
    train_xvec = { key.strip():vec.tolist() for key,vec in read_vec_flt_ark('../../../data/xvec_v2_train.ark')}
    
    assert(len(list(train_xvec.keys()))==1092009)

    trainval_spk2utt = {line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open('../../../data/spk2utt_train','r').readlines()}
    assert(len(list(trainval_spk2utt.keys()))==5994)
    
    train_spk2utt = {spk:trainval_spk2utt[spk] for spk in train_voice_list}
    valid_spk2utt = {spk:trainval_spk2utt[spk] for spk in valid_voice_list}
    
    ## For Test data
    test_xvec = { key.strip():vec.tolist() for key,vec in read_vec_flt_ark('../../../data/xvec_v2_test.ark')}
    assert(len(list(test_xvec.keys()))==36237)
    test_spk2utt = {line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open('../../../data/spk2utt_test','r').readlines()}
    assert(len(list(test_spk2utt.keys()))==118)


    train_dataset = EmbedDataset(face_embed_data, train_face_list, train_xvec, train_voice_list,train_spk2utt)
    valid_dataset = EmbedDataset(face_embed_data, valid_face_list, train_xvec, valid_voice_list,valid_spk2utt)
    test_dataset = EmbedDataset(face_embed_data, test_face_list, test_xvec, test_voice_list,test_spk2utt)
    print('Creating loaders with batch_size as ', bs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4,
                                               collate_fn=custom_collate, pin_memory= True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4,
                                               collate_fn=custom_collate, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=3,
                                              collate_fn=custom_collate, pin_memory = True)
    
    print("Time taken for data loading: ", time.time() - start)
    return train_loader,valid_loader,test_loader
