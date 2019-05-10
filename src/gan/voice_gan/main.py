
import os
import random
import argparse
from tqdm import tnrange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from data_loader import train_loader
from model import *
from config import args_parser

if __name__ == '__main__':
    # parse args
    opt = args_parser()

    # output paths
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    # parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    scale_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2)


    netG = Custom_generator(ngpu).to(device)
    netG.apply(weights_init)

    # Load pretrained joint embedding weights

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    trained_epoch = 0
    real_label = 1
    fake_label = 0
    flag = 1
    G_losses=[]
    D_losses=[]

    # Fixed Noise for visualization
    fixed_noise = torch.randn(opt.bs, nz // 2, 1, 1, device=device)

    # Load from previous checkpoint
    if opt.load != '':
        checkpoint = torch.load(opt.load)
        trained_epoch = checkpoint['epoch']
        netD.load_state_dict(checkpoint['discriminator'])
        netG.load_state_dict(checkpoint['generator'])

    "==========================================================================="
    # Training
    for epoch in tnrange(trained_epoch+1, trained_epoch+opt.nepoch+1):
        for i, batch in enumerate(train_loader, 0):
            batch_voice, face_real, gold_labels = batch
            batch_voice, gold_labels = batch_voice.to(device), gold_labels.to(device)

            if flag and i == 0:
                fixed_voice = batch_voice
                flag = 0

            "Udpate D-Net: maximize log(D(x)) + log(1 - D(G(z)))"

            "-----------Train with ALL-REAL batch-----------------"
            netD.zero_grad()

            real_cpu = face_real.to(device)
            b_size = real_cpu.size(0)

            label = torch.full((b_size,), real_label, device=device)

            # Downsample the input image
            # down_real_cpu = nn.AvgPool2d(2, stride=2)(real_cpu)args =

            output = netD(real_cpu).view(-1)  # Forward pass real batch through D
            errD_real = criterion(output, label)  # Calculate loss on all-real batch

            errD_real.backward()
            D_x = output.mean().item()

            # -----------Train with ALL-FAKE batch-----------------

            # Generate batch of latent vectors
            gaussian_noise = torch.randn(b_size, nz//2, 1, 1, device=device)

            fake = netG(gaussian_noise, batch_voice)  # Generate fake image batch
            label.fill_(fake_label)
            #down_fake = nn.AvgPool2d(2, stride=2)(fake.clone().detach())

            output = netD(fake.detach()).view(-1)         # Classify all fake batch with D
            errD_fake = criterion(output, label)  # D's loss on the all-fake batch

            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

    #         mse_loss = nn.MSELoss(reduction='mean')(down_real_cpu.clone(), down_fake.clone())/(32**2)
    #         loss = lambda1*(errD) + lambda2*mse_loss
    #         loss.backward()

            optimizerD.step()          # Update D

            "----------------------------------------------------------------------"
            "Udpate G-NET: maximize log(D(G(z)))"

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            errG = criterion(output, label)  # G's loss based on all-fake batch output of updated D

            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()  # Update G
            if i % 600 ==0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, trained_epoch + opt.nepoch, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % opt.outf,
                                  normalize=True)
                fake = netG(fixed_noise, fixed_voice)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                                  normalize=True)

        # do checkpointing
        if epoch % 3 == 0:
            print("Saving model ...")
            state_dict = {'epoch': epoch+1,
                          'generator': netG.state_dict(),
                          'discriminator': netD.state_dict()
                          }

            save_path = "saved_models/ck_{}.pth.tar".format(epoch+1)
            torch.save(state_dict, save_path)

    fig=plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Losses")
