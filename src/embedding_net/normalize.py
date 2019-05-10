import numpy as np
import torch
from torch import nn
import sys
import re
import struct,logging
import itertools
import torchvision
import pandas as pd
import json
import numpy as np 
import sys
import re
import struct
import subprocess
from subprocess import Popen
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
def load_json(filename='data.json'):
    with open(filename, 'r') as outfile:
        data = json.load(outfile)
    return data


def write_to_json(data, filename='data.json'):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
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
#print(len(train_voice_list),"SPK@UTT",len(list(trainval_spk2utt.keys())))
count=0
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
    
full_data =[]
for i in list(train_xvec.keys()):
    full_data.append(train_xvec[i])
full_data = np.array(full_data)

test_xvec = { key.strip():vec.tolist() for key,vec in read_vec_flt_ark('../../../data/xvec_v2_test.ark')}
assert(len(list(test_xvec.keys()))==36237)
test_spk2utt = {line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open('../../../data/spk2utt_test','r').readlines()}
assert(len(list(test_spk2utt.keys()))==118)

test_data = []
for i in list(test_xvec.keys()):
    test_data.append(test_xvec[i])
test_data = np.array(test_data)

total_data = np.concatenate((full_data,test_data),axis=0)
mx = np.max(total_data,axis=0)
mn = np.min(total_data,axis=0)
data = -1 +(2*(full_data - mn)) / (mx-mn)
new_test = -1 +(2*(test_data - mn)) / (mx-mn)

norm_train={}

for e,i in enumerate(list(train_xvec.keys())):
    norm_train[i]=data[e].tolist()
norm_test ={}

for e,i in enumerate(list(test_xvec.keys())):
    norm_test[i]=new_test[e].tolist()

write_to_json(norm_train,'../../../data/norm_train_xvecs.json')
write_to_json(norm_test,'../../../data/norm_test_xvecs.json')
