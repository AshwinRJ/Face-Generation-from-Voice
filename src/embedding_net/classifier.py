#! /bin/python
import numpy as np
import torch
import torch.nn as nn
import os
import time
import itertools

class Classifier(nn.Module):
    def __init__(self,hiddens=[300,150,50],num_classes=6112):
        self.speech_projection = nn.Linear(self.speech_embed_dim,hiddens[0])
        self.image_projection = nn.Linear(self.face_embed_dim,hiddens[0])
        self.layers = []
        for i in range(len(self.hidden_dims)-1):
            self.layers.append(nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
        self.face_net = nn.Sequential(*[self.face_projection]+[self.layers])
        self.voice_net = nn.Sequential(*[self.speech_projection]+[self.layers])
        self.projection = nn.Linear(50,num_classes)
        print("Initialized Model")

    def forward(self,embedding,face=True):
        if face:
            embedding = self.face_net(embedding)
        else
            embedding = self.voice_net(embedding)
        output = self.projection(embedding)
        return output


