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

""""
===============================================================================================
"""""
class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, face_data, face_list, voice_data, voice_list,spk2utt):
        self.face_data = face_data
        self.face_list = face_list
        self.voice_data = voice_data
        self.voice_list = voice_list
        self.spk2utt = spk2utt
        self.num_class = len(voice_list)
        self.num_face_repeats = 30 ##Number of times I go through examples 
        self.num_voice_repeats = 30
        #print(list(self.spk2utt.keys()))
        #self.face_range = np.arange(self.num_face_repeats)
        #self.voice_range = np.arange(self.num_voice_repeats)
        #self.combo_list = list(itertools.product(self.face_range,self.voice_range))
        #self.class_repeat_num = np.zeros((self.num_class,2),dtype=np.int32)
        #print('Number of classes is',self.num_class)

    def __len__(self):
        return self.num_class * (self.num_face_repeats * self.num_voice_repeats)


    def __getitem__(self, index):
        #new_index = index
        #print(index)
        index = int(index % self.num_class)
        #print("TWO",index)
        assert self.face_list[index] in list(self.face_data.keys())
        assert self.voice_list[index] in list(self.spk2utt.keys())
        #print(self.face_list[index],list(self.face_data.keys()))
        face_id = self.face_list[index]
        voice_id = self.voice_list[index]
        #print("VOICE ID  and LEN SPK2UTT:",voice_id,len(self.spk2utt[voice_id]))
        #repeat_index = self.class_repeat_num[index]
        #print(self.combo_list[new_index % self.num_class])
        i = np.random.randint(low=0,high=min(self.num_face_repeats,len(self.face_data[face_id])))
        j = np.random.randint(low=0,high=min(self.num_voice_repeats,len(self.spk2utt[voice_id])))
        face_embed = torch.Tensor(np.array(self.face_data[face_id][i]))
        voice_embed = torch.Tensor(np.array(self.voice_data[self.spk2utt[voice_id][j]]))
        #print(np.array(self.face_data[face_id][i]).shape,np.array(self.voice_data[self.spk2utt[voice_id][j]]).shape)
        assert(face_embed.size()[0]==512 and voice_embed.size()[0]==512)
        concatenated_feat_vec = torch.cat((face_embed, voice_embed),0)
        assert(concatenated_feat_vec.size()[0]==1024)
        #print(concatenated_feat_vec.size()[0],'CFeat Size')
        #if concatenated_feat_vec==torch.zeros((1024)):
        #    print('Sending 0 input')
        return concatenated_feat_vec
"""
class EmbedLoader(torch.utils.data.DataLoader):
   # def __init__(self,Dataset,batch_size,shuffle=True):
        super(EmbedLoader).__init__()
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.dataset=dataset
        self.num_face_repeats = 50 ##Number of times I go through examples 
        self.num_voice_repeats = 22
        self.face_range = np.arange(self.num_face_repeats)
        self.voice_range = np.arange(self.num_voice_repeats)

   # def __iter__(self):
        for i in range()
            selected_face_indices = np.random.choice(face_array[i],50)
            selected_voice_indices = np.random.choice(face_array[i],50)

"""



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

    valid_face_list = train_face_list[-200:]
    train_face_list = train_face_list[:-200]
    valid_voice_list = train_voice_list[-200:]
    train_voice_list = train_voice_list[:-200]
    train_xvec = load_json("../../../data/norm_train_xvecs.json") #{ key.strip():vec.tolist() for key,vec in read_vec_flt_ark('../../../data/xvec_v2_train.ark')}
    assert(len(list(train_xvec.keys()))==1092009)

    trainval_spk2utt = {line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open('../../../data/spk2utt_train','r').readlines()}
    assert(len(list(trainval_spk2utt.keys()))==5994)
    train_spk2utt = {spk:trainval_spk2utt[spk][:-20] for spk in train_voice_list}
    valid_spk2utt = {spk:trainval_spk2utt[spk][-20:] for spk in valid_voice_list}
    ## For Test data
    test_xvec = load_json("../../../data/norm_test_xvecs.json")#{ key.strip():vec.tolist() for key,vec in read_vec_flt_ark('../../../data/xvec_v2_test.ark')}
    assert(len(list(test_xvec.keys()))==36237)
    test_spk2utt = {line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open('../../../data/spk2utt_test','r').readlines()}
    assert(len(list(test_spk2utt.keys()))==118)


    """
    # Dummy voice data for now:
    ## SANITY CHECK CODE
    voice_embed_data = {}
    face_embed_data = {}
    voice_list = train_voice_list + valid_voice_list
    face_list = train_face_list + valid_face_list
    print('The dataset has ',len(voice_list), 'elements totally')

    for k, i_d in enumerate(voice_list):
        voice_embed_data[i_d] = np.random.randn(50,512)
        face_embed_data[face_list[k]] = np.random.randn(50,512)
 """
    train_dataset = EmbedDataset(face_embed_data, train_face_list, train_xvec, train_voice_list,train_spk2utt)
    valid_dataset = EmbedDataset(face_embed_data, valid_face_list, train_xvec, valid_voice_list,valid_spk2utt)
    test_dataset = EmbedDataset(face_embed_data, test_face_list, test_xvec, test_voice_list,test_spk2utt)
    print('Creating loaders with batch_size as ', bs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory= True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=3, pin_memory = True)
    print("Time taken for data loading: ", time.time()-start)
    return train_loader,valid_loader,test_loader

