#! /bin/python
import numpy as np
import torch
import torch.nn as nn
import os
import time
import itertools
## MODEL ARCHITECTURE- MLP with N-Pair Loss


class Classifier(nn.Module):
    def __init__(self,hidden_dims=[300,150,50],num_classes=5992):
        super(Classifier,self).__init__()
        self.speech_embed_dim = 512
        self.face_embed_dim = 512
        self.hidden_dims = hidden_dims
        self.layers=[]
        self.speech_projection = nn.Linear(self.speech_embed_dim,hidden_dims[0])
        ##self.speech_projection = nn.Linear(self.speech_embed_dim,self.hidden_dims[0])
        ##self.image_projection = nn.Linear(self.face_embed_dim,self.hidden_dims[0])
        for i in range(len(self.hidden_dims)-1):
            self.layers.append(nn.Linear(self.hidden_dims[i],self.hidden_dims[i+1]))
            if i < len(self.hidden_dims)-2:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Hardtanh())
            #self.layers.append(nn.Dropout(dropout_prob))
        self.model = nn.Sequential(*self.layers)
        print(self.model)
        print("Initialized Model")

    def forward(self,voice,faces=None):
        #print("Started Forward",voice.size())
        projection = self.speech_projection(voice)
        speech = self.model(projection)
        if faces is not None:
            # print('I have faces',faces.size())
            projection2 = self.speech_projection(faces)
            faces = self.model(projection2)
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
        # print("Initilized Loss")

    def __call__(self,voice_embeds,face_embeds):
        # print("Called Loss")
        return self.forward(voice_embeds,face_embeds)

    def forward(self, voice_embeds, face_embeds):
        # print("Loss Forward activated")
        lst= time.time()
        self.N = len(voice_embeds)
        #anchor_indices = torch.arange(len(voice_embeds))
        #positive_indices = torch.arange(len(face_embeds))
        array= np.arange(self.N)
        # print('Starting listcomp')
        st=time.time()
        negative_indices = np.array([np.delete(array, i) for i in range(len(array))])       
        #print('Generated neg indices in',time.time()-lst)
        # print('Done listcomp in time',time.time()-st, 'NI shape', negative_indices.shape)  
        anchors = voice_embeds    # (n, embedding_size)
        positives =   face_embeds #embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = torch.stack([face_embeds[negative_indices[i]] for i in range(self.N)], dim=0) # (n, n-1, embedding_size)
        #print('Generated negatives in',time.time()-lst)
        # print('Negatives shape:',negatives.size())
        #print(self.n_pair_loss(anchors,positives,negatives),anchors,negatives,positives)
        #sys.exit(0)
        loss = self.n_pair_loss(anchors, positives, negatives) + self.l2_reg*self.l2_loss(anchors,positives)
        #print('Got loss in',time.time()-lst)
        del negatives,positives,anchors 
        return loss

    def n_pair_loss(self,anchors, positives, negatives):
        """
        Calculates N-Pair loss between anchor positive and negative examples
        anchors: A torch.Tensor, (n, embedding_size)
        positives: A torch.Tensor, (n, embedding_size)
        negatives: A torch.Tensor, (n, n-1, embedding_size)
        Loss: A scalar
        """
        eps= 1e-10
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)
        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(eps+1+x))
        return loss
