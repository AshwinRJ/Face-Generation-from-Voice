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


class ImageLoader(Dataset):
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

        i, j = torch.randint(low=0, high=49, size=(1, 1)).item(
        ), torch.randint(low=0, high=49, size=(1, 1)).item()

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
face_list = [item[:-1] for item in common_meta['VGGFace2 ID '].tolist()]
voice_list = [item[:-1] for item in common_meta['VoxCeleb2 ID '].tolist()]

# Dummy voice data for now:
voice_embed_data = {}
for k, i_d in enumerate(voice_list):
    voice_embed_data[i_d] = face_embed_data[face_list[k]]

dataset = ImageLoader(face_embed_data, face_list, voice_embed_data, voice_list, repeats=1)
loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, num_workers=8)


# Example Train Loop

num_epochs = 2
repeat_factor = 4  # num_repeats_over_data

for epoch in range(num_epochs):
    for r in range(repeat_factor):
        for embed in loader:

            # Todo

            pass
        print("Epoch: {:02} | Num of Pass over data {}".format(epoch, r))
