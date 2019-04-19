#! /bin/python

import torch
import torch.nn as nn
import os
import itertools
## MODEL ARCHITECTURE
## MLP with 3 units


class EmbeddingNet(nn.Module):
    def __init__(self,hidden_dims=[2048,2048,1024,128],dropout_prob=0.4):
        super(EmbeddingNet,self).__init__()
        self.speech_embed_dim = 512
        self.face_embed_dim = 512
        self.hidden_dims = hidden_dims
        self.layers=[]
        ##self.speech_projection = nn.Linear(self.speech_embed_dim,self.hidden_dims[0])
        ##self.image_projection = nn.Linear(self.face_embed_dim,self.hidden_dims[0])
        for i in range(len(self.hidden_dims)-1):
            self.layers.append(nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1]))
            self.layers.append(nn.Dropout(dropout_prob))
        self.hiddens = nn.Sequential()
    
    def forward(self,voice,faces=None):
        speech = self.hiddens(voice)
        if faces is not None:
            face = self.hiddens(face)
        return speech,face


class NPairLoss():
    """
    Takes in face embeddings of (bs,embed_size) and Voice embeddings of (bs,embed_size)

    Here, in our experiments N=bs
    Computes N-Loss in the DML paradigm

    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, l2_penalty=0.02):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_penalty

    def forward(self, voice_embeds, face_embeds,embeddings, target):
        self.N = len(voice_embeds)
        #anchor_indices = torch.arange(len(voice_embeds))
        #positive_indices = torch.arange(len(face_embeds))
        negative_indices = torch.from_numpy(itertools.permutations(np.arange(self.N)))
        anchors = voice_embeds    # (n, embedding_size)
        positives =   face_embeds #embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = [torch.cat(face_embeds[negative_indices[i]],dim=0) for i in range(N)]
        #negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        loss = self.n_pair_loss(anchors, positives, negatives) + self.l2_reg * self.l2_loss(anchors, positives)

        return loss

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss between anchor positive and negative examples
        anchors: A torch.Tensor, (n, embedding_size)
        positives: A torch.Tensor, (n, embedding_size)
        negatives: A torch.Tensor, (n, n-1, embedding_size)
        Loss: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)
        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

"""
    def l2_loss(anchors, positives):
        
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]
"""