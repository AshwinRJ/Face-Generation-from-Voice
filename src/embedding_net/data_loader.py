import os
import json
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
torch.backends.cudnn.bencmark = True

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

device = "cuda" if torch.cuda.is_available() else "cpu"


"""========================================================================="""
# Helper Functions


def load_json(filename='data.json'):
    with open(filename, 'r') as outfile:
        data = json.load(outfile)
    return data


def write_to_json(data, filename='data.json'):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


class EmbedLoader(Dataset):
    def __init__(self, face_data, face_list, voice_data, voice_list, repeats):
        self.face_data = face_data
        self.face_list = face_list

        self.voice_data = voice_data
        self.voice_list = voice_list

        self.num_class = len(voice_list)
        self.repeats = repeats

    def __len__(self):
        return self.num_class

    def __getitem__(self, index):

        i, j = torch.randint(low=0, high=49, size=(1, 1)).item(), torch.randint(low=0, high=49, size=(1, 1)).item()

        face_embed = torch.tensor(self.face_data[self.face_list[index]][i])
        voice_embed = torch.tensor(self.voice_data[self.voice_list[index]][j])

        return torch.cat((face_embed, voice_embed), dim=0)


"""========================================================================="""


# Load the data files
common_meta = pd.read_csv("data/vox2_meta.csv")
face_embed_data = load_json("vggface2_voxceleb2_embeddings.json")
# voice_embed_data = load_json("vggface2_voxceleb2_embeddings.json")

# List of face and voice IDs
# Contains the class names
dont_include = ['n003729 ' , 'n003754 ']

train_face_list, valid_face_list = [], []
train_voice_list, valid_voice_list = [], []

for i in range(len(common_meta['Set '])):
   if common_meta['Set '].iloc[i] == "dev " and common_meta['VGGFace2 ID '].iloc[i] not in dont_include:
       train_face_list.append(common_meta['VGGFace2 ID '].iloc[i][:-1])
       train_voice_list.append(common_meta['VoxCeleb2 ID '].iloc[i][:-1])

   elif common_meta['Set '].iloc[i] == "test " and common_meta['VGGFace2 ID '].iloc[i][:-1] not in dont_include:
       valid_face_list.append(common_meta['VGGFace2 ID '].iloc[i][:-1])
       valid_voice_list.append(common_meta['VoxCeleb2 ID '].iloc[i][:-1])

# Dummy voice data for now:
voice_embed_data = {}
for k, i_d in enumerate(voice_list):
    voice_embed_data[i_d] = face_embed_data[face_list[k]]

train_dataset = EmbedLoader(face_embed_data, train_face_list, voice_embed_data, train_voice_list)
valid_dataset = EmbedLoader(face_valid_data, valid_face_list, voice_embed_data, valid_voice_list)


