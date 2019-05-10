#! /bin/python

import json
from PIL import Image
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_path = "../../../../data/vgg_2000/"


def load_json(filename='data.json'):
    with open(filename, 'r') as outfile:
        data = json.load(outfile)
    return data


def write_to_json(data, filename='data.json'):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


class GANDL(Dataset):
    def __init__(self, tuple_set, spk_feats):
        self.tuple_set = tuple_set
        self.voice_embeds = spk_feats

    def __len__(self):
        return len(self.tuple_set)

    def __getitem__(self, index):
        utt_id, face_path, label = self.tuple_set[index][0], self.tuple_set[index][1], self.tuple_set[index][2]
        voice_embed = torch.from_numpy(np.array(self.voice_embeds[utt_id])).float()
        face = Image.open(base_path+face_path)
        face = np.array(face.resize((64, 64)))
        face = torchvision.transforms.ToTensor()(face)
        labels = torch.tensor(label).long()

        return voice_embed, face, labels

# Read voice_utt_ids


norm_xvec = load_json('../../../../data/norm_train_xvecs.json')
tuples = load_json('../../../../data/2000spk_tuples.json')

train_set = GANDL(tuples, norm_xvec)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=True, num_workers=6)
