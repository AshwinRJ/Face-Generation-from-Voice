#! /bin/python
import os
import torch
import torch.nn as nn
from model import EmbeddingNet, NPairLoss
from data_loader import train_dataset,valid_dataset


torch.backends.cudnn.benchmark=True
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Device is',device)
expt_prefix="v1"
tlog, vlog = SummaryWriter("./"+expt_prefix+"logs/train_pytorch"), SummaryWriter("./"+expt_prefix+"logs/val_pytorch")
load_path=expt_prefix+"logs/"
lp = open("./"+expt_prefix+"log","w+") ## Output log file

class TrainValidate():

    def __init__(self,hiddens=[2048,2048,1024,128],num_epochs=50,initial_lr=1e-3,batch_size=128,weight_decay=0,load=False):
        self.num_epochs = num_epochs
        self.bs = batch_size
        self.criterion = NPairLoss()
        self.net = EmbeddingNet(hiddens).to(device)
        self.embed_size = 512
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=8)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.bs, shuffle=False, num_workers=8)
        self.train_loss = 0
        self.valid_loss = 0
        self.patience = 5
        self.lr = initial_lr
        self.optimizer =  torch.optim.Adam(lr=self.lr,weight_decay=weight_decay)
        self.init_epoch = 0
        self.net.apply(self.weights_init)
        if load:
            print('Loading model from past')
            #self.net.load_state_dict(torch.load(load_path))
            self.init_epoch=self.load(load_path)         
        lp.write(self.net())
        self.run()
  

    def train_validate(self,train_loader):
        for epoch in range(self.init_epoch,self.num_epochs+self.init_epoch):
            tstart=time.time()
            self.train_loss,train_acc=self.run_epoch(train_loader) 
            print("-------------------------------------------------------------------------------------------")
            print("Processing epoch "+str(epoch))
            tlog.add_scalar('Train Loss',self.train_loss)
            print('Training Loss is ',self.train_loss, ' Learning Rate is ',self.get_lr())
            self.eval_loss,eval_acc=self.run_epoch(dev_loader,False)
            vlog.add_scalar('Validation Loss'+epoch,self.eval_loss)
            print('Validation Loss is ',self.eval_loss)
            tend=time.time()
            print('Epoch ',str(epoch), ' was done in ',str(tend-tstart),' seconds')
            print("-------------------------------------------------------------------------------------------")
            for tag, value in self.net.named_parameters():
                tag = tag.replace('.', '/')
                tlog.add_histogram(tag, value.data,epoch)
                tlog.add_histogram(tag+'/grad', value.data,epoch)
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
            

    def run_epoch(self,loader,update=False):
        epoch_loss = 0
        for batch_index, (embedding) in enumerate(train_loader):
            face_embedding = embedding [:self.embed_size]
            voice_embedding = embedding[self.embed_size:]
            output_face_embed = self.net(face_embedding)
            output_voice_embed = self.net(voice_embedding)
            loss = self.criterion(output_face_embed,output_voice_embed)
            epoch_loss += loss.item()
            if update:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return epoch_loss

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
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias,0)
    
    def test(self,embedding):
        return self.net(embedding).cpu().numpy()

    def run(self):
        self.train_validate()
        self.test()

if __name__ == "__main__":
    import multiprocessing
    TrainValidate()
    



