#! /bin/python
import os
import torch
import torch.nn as nn
from dl_speech import get_data_loaders
from classifier import Classifier
from tensorboardX import SummaryWriter
import time
import numpy as np 
torch.backends.cudnn.benchmark=True
import sys
#from dl_speech import write_to_json,train_xvecs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
expt_prefix="v4"
tlog, vlog = SummaryWriter("../../"+expt_prefix+"logs/train_pytorch"), SummaryWriter("../../"+expt_prefix+"logs/val_pytorch")
load_path="v4logs/model_dict30.pt"
lp = open("./"+expt_prefix+"log","w+") ## Output log file

class TrainValidate():


<<<<<<< HEAD
    def __init__(self,hiddens=[300,150,50],num_epochs=100,initial_lr=1e-3,batch_size=3500,weight_decay=1e-3,load=False):
=======
    def __init__(self,hiddens=[256,128,50],num_epochs=2000,initial_lr=1e-3,batch_size=3500,weight_decay=1e-3,load=False):
>>>>>>> 42b8df987b495858f58244c8e47fd1af18c8973f
        self.num_epochs = num_epochs
        self.bs = batch_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.net = Classifier().to(device)
        #self.net = torch.nn.DataParallel(self.net,device_ids=[0,1,2,3])
        self.embed_size = 512
        #self.train_loader, self.valid_loader = get_data_loaders()
        self.train_loss = 0.0
        self.valid_loss = 0.0
        self.patience = 100
        self.lr = initial_lr
        self.optimizer =  torch.optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=weight_decay)
        self.init_epoch = 0
        self.net.apply(self.weights_init)
        if load:
            print('Loading model from past')
            self.init_epoch=self.load(load_path)
            self.test(train_xvecs)
        #lp.write(expt_prefix+' Model with hiddens '+str(hiddens)+'\n\n')
        #self.run()
  

    def train_validate(self):
        print('Batch size is',self.bs)
<<<<<<< HEAD
        train_loader,valid_loader, _ = get_data_loaders(self.bs)
=======
        train_loader,valid_loader,_ = get_data_loaders(self.bs)
>>>>>>> 42b8df987b495858f58244c8e47fd1af18c8973f
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=self.patience,min_lr=1e-7)
        best_loss= np.inf
        for epoch in range(self.init_epoch,self.num_epochs+self.init_epoch):
            tstart=time.time()
            print("-------------------------------------------------------------------------------------------")
            print("Processing epoch "+str(epoch))
            self.train_loss,train_acc=self.run_epoch(train_loader) 
            tlog.add_scalar('Train Loss', self.train_loss)
<<<<<<< HEAD
            print('Training Loss is ', self.train_loss, "Training Accuracy",train_acc)
            self.eval_loss,valid_acc=self.run_epoch(valid_loader,False)
            vlog.add_scalar('Validation Loss'+ str(epoch), self.eval_loss)
            print('Validation Loss is ',self.eval_loss,"Validation Accuracy",valid_acc)
=======
            print("Training Loss: {:0.5f} | Training Acc: {:0.4f} | Face Acc: {:0.4f} | Voice Acc: {:0.4f} | LR: {:0.5f}".format(self.train_loss, train_acc,ftacc,vtacc,self.get_lr()))
            self.eval_loss, fvacc, vvacc = self.run_epoch(valid_loader,False)
            valid_acc = 0.5 * (fvacc + vvacc)
            vlog.add_scalar('Validation Loss'+ str(epoch), self.eval_loss)
            print("Validation Loss: {:0.5f} | Validation Acc: {:0.4f} | Val Face Acc: {:0.4f} | Val Voice Acc: {:0.4f}".format(self.eval_loss, valid_acc,fvacc,vvacc))
>>>>>>> 42b8df987b495858f58244c8e47fd1af18c8973f
            tend=time.time()
            print('Epoch ',str(epoch), ' was done in ',str(tend-tstart),' seconds')
            print("-------------------------------------------------------------------------------------------")
            for tag, value in self.net.named_parameters():
                tag = tag.replace('.', '/')
                tlog.add_histogram(tag,value.data,global_step=epoch)
                tlog.add_histogram(tag+'/grad',value.data,global_step=epoch)
            if self.eval_loss < best_loss:
                best_loss = self.eval_loss
                pat = 0
                self.save(epoch)
            else:
                pat += 1
                if pat >= self.patience:
                    print("Early stopping !")
                    break
            sch.step(self.eval_loss)

        print('Training and Validation complete !')

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def save(self,epoch):
        torch.save({'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_loss,
            'dev_loss':self.eval_loss},"./"+expt_prefix+"logs/model_dict"+str(epoch)+".pt")
            

    def run_epoch(self,loader,update=True):
        epoch_face_loss = 0
        epoch_voice_loss = 0
        start_time = time.time()
        vaccu = 0
        for batch_index,(voice_batch,class_labels) in enumerate(loader):
            self.optimizer.zero_grad()
           # print('Embedding size',embedding.size())
<<<<<<< HEAD
            voice_batch = voice_batch.to(device)
            class_labels = class_labels.to(device)
            voice_logits = self.net(voice_batch)
            loss = self.criterion(voice_logits,class_labels)
            epoch_loss += loss.item()
            vop = torch.nn.functional.softmax(voice_logits,dim=1)
            _,vpred=torch.max(vop,1)
            vacc = float(torch.sum(torch.flatten(vpred==class_labels)).item() / len(voice_batch))
            vaccu+=vacc
=======
            face_batch = torch.FloatTensor(face_batch).to(device)
            voice_batch = torch.FloatTensor(voice_batch).to(device)
            class_labels = torch.LongTensor(class_labels).to(device)
            face_logits = self.net(face_batch,face=True)
             ##Net takes voice, faces
            loss = self.criterion(face_logits,class_labels)
            epoch_face_loss += loss.item()
            fop = torch.nn.functional.softmax(face_logits,dim=1)
            _,fpred=torch.max(fop,1)
            facc=torch.sum(torch.flatten(fpred==class_labels))
            faccu+= facc.item() / len(face_logits)
            if update:
                loss.backward()
                #self.optimizer.step()
            voice_logits = self.net(voice_batch,face=False)
            loss = self.criterion(voice_logits,class_labels)
            epoch_voice_loss += loss.item()
            vop = torch.nn.functional.softmax(voice_logits,dim=1)
            _,vpred=torch.max(vop,dim=1)
            vacc=torch.sum(torch.flatten(vpred==class_labels))
            vaccu+=vacc.item() / len(voice_batch)
>>>>>>> 42b8df987b495858f58244c8e47fd1af18c8973f
            if update:
                loss.backward()
                self.optimizer.step()
        torch.cuda.empty_cache()
<<<<<<< HEAD
        epoch_loss /= (batch_index+1)
        vaccu /= (batch_index +1)
        return epoch_loss,vaccu
=======
        epoch_voice_loss /= (batch_index+1)
        epoch_face_loss /= (batch_index+1)
        faccu /= (batch_index +1)
        vaccu /= (batch_index +1)
        epoch_loss = 0.5*(epoch_voice_loss + epoch_face_loss)
        return epoch_loss,faccu,vaccu
>>>>>>> 42b8df987b495858f58244c8e47fd1af18c8973f

    def load(self,path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.train_loss= checkpoint['loss']
        self.eval_loss=checkpoint['dev_loss']
        return epoch

    def weights_init(self,m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)      
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias,0)
    
    def test(self,train_xvecs):
        res={}
        for i,key in enumerate(list(train_xvecs.keys())):
            voice_feat = train_xvecs[i].to(device)
            output = self.net(voice_feat,test=True)
            res[key] = output.detach().cpu().numpy().tolist()
        write_to_json(res,'voicenw_embeddings.json')

    def run(self):
        self.train_validate()
        #self.test()

if __name__ == "__main__":
    import multiprocessing
    print('Device is',device)
    TrainValidate()


    



