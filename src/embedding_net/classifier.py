#! /bin/python
import numpy as np
import torch
import torch.nn as nn
import os
import time
import itertools

class Classifier(nn.Module):
    def __init__(self,hiddens=[300,150,50],num_classes=5794):
        super(Classifier,self).__init__()
        self.speech_embed_dim = 512
        self.face_embed_dim = 512
        self.speech_projection = nn.Linear(self.speech_embed_dim,hiddens[0])
        self.image_projection = nn.Linear(self.face_embed_dim,hiddens[0])
        self.layers = []
        self.hidden_dims = hiddens
        for i in range(len(self.hidden_dims)-1):
            self.layers.append(nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(50,num_classes))
        self.model = nn.Sequential(*self.layers)
        print("Initialized Model")

    def forward(self,embedding,face=True):
        if face:
            embedding = self.image_projection(embedding)
        else:
            embedding = self.speech_projection(embedding)
        output = self.model(embedding)
        return output


