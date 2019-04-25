#! /bin/python
import numpy as np
import torch
import torch.nn as nn
import os
import itertools
## MODEL ARCHITECTURE- MLP with N-Pair Loss


class EmbeddingNet(nn.Module):
    def __init__(self,hidden_dims=[512,512,256,128,50],dropout_prob=0.4):
        super(EmbeddingNet,self).__init__()
        self.speech_embed_dim = 512
        self.face_embed_dim = 512
        self.hidden_dims = [512] + hidden_dims
        self.layers=[]
        ##self.speech_projection = nn.Linear(self.speech_embed_dim,self.hidden_dims[0])
        ##self.image_projection = nn.Linear(self.face_embed_dim,self.hidden_dims[0])
        for i in range(len(self.hidden_dims)-1):
            self.layers.append(nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1]))
            self.layers.append(nn.Dropout(dropout_prob))
        self.hiddens = nn.Sequential(*self.layers)
        print("Initialized Model")

    def forward(self,voice,faces=None):
        #print("Started Forward",voice.size())
        speech = self.hiddens(voice)
        if faces is not None:
            print('I have faces',faces.size())
            faces = self.hiddens(faces)
        #print('Done forward',speech.size(),faces.size())
        return speech, faces


class NPairLoss():
    """
    Takes in face embeddings of (bs,embed_size) and Voice embeddings of (bs,embed_size)

    Here, in our experiments N=bs
    Computes N-Loss in the DML paradigm
    Idea from https://dl.acm.org/citation.cfm?id=3240601 - Face Voice Matching Using Cross-Modal Embeddings 
    Original Paper- Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_penalty=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_penalty
        self.l2_loss = torch.nn.MSELoss()
        print("Initilized Loss")

    def __call__(self,voice_embeds,face_embeds):
        print("Called Loss")
        return self.forward(voice_embeds,face_embeds)

    def forward(self, voice_embeds, face_embeds):
        print("Loss Forward activated")
        self.N = len(voice_embeds)
        #anchor_indices = torch.arange(len(voice_embeds))
        #positive_indices = torch.arange(len(face_embeds))
        negative_indices = torch.from_numpy(np.array(list(itertools.permutations(np.arange(self.N))))
        anchors = voice_embeds    # (n, embedding_size)
        positives =   face_embeds #embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = torch.tensor([torch.cat(face_embeds[negative_indices[i]],dim=0) for i in range(N)]) # (n, n-1, embedding_size)
        print('Negatives shape:',negatives.size())

        loss = self.n_pair_loss(anchors, positives, negatives) +self.l2_reg*self.l2_loss(anchors,positives)
        return loss

    def n_pair_loss(self,anchors, positives, negatives):
        """
        Calculates N-Pair loss between anchor positive and negative examples
        anchors: A torch.Tensor, (n, embedding_size)
        positives: A torch.Tensor, (n, embedding_size)
        negatives: A torch.Tensor, (n, n-1, embedding_size)
        Loss: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        assert(anchors.size() == (self.N, 1, 512))
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)
        assert(positives.size() == (self.N, 1, 512))
        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss
