
import os
import random
from tqdm import tnrange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from data_loader import train_loader
from model import *
from config import args_parser

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    # parse args
    opt = args_parser()
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    # Models
    # Sphere face
    sphereface = sphere20a()
    sphereface.load_state_dict(torch.load('sphereface/model/sphere20a_20171020.pth'))
    sphereface.to(device)
    sphereface.eval()

    # Generator and Discriminator
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)

    netG = Custom_generator(ngpu, joint_hiddens=[300, 150, 50]).to(device)
    netG.apply(weights_init)

    if opt.load_joint_embeddings:
        # hidden_dims of pre-trained layers -->  joint_hiddens=[300, 150, 50]
        # Load pretrained joint embedding weights in the generator
        joint_net_ck = torch.load(opt.joint_net_path)
        pretrained_dict = joint_net_ck['model_state_dict']

        model_dict = netG.Joint_embedding_net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        netG.Joint_embedding_net.load_state_dict(pretrained_dict)

    # Criterion
    criterion = nn.BCELoss()
    cosine_criterion = nn.CosineEmbeddingLoss()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # Training parameters
    trained_epoch = 0
    flag = True
    real_label = 1
    fake_label = 0
    G_losses, D_losses = [], []

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

            if flag:
                vutils.save_image(real_batch, '{}/{}_real.png'.format(
                                    opt.outf, opt.save_name), normalize=True)
                fixed_voice = batch_voice
                flag = False

            "Udpate D-Net: maximize log(D(x)) + log(1 - D(G(z)))"

            "-----------Train with ALL-REAL batch-----------------"
            netD.zero_grad()

            real_batch = face_real.to(device)
            b_size = real_batch.size(0)

            label = torch.full((b_size,), real_label, device=device)

            output = netD(real_batch).view(-1)  # Forward pass real batch through D
            errD_real = criterion(output, label)  # Calculate loss on all-real batch

            errD_real.backward()
            D_x = output.mean().item()

            # -----------Train with ALL-FAKE batch-----------------

            # Generate batch of latent vectors
            gaussian_noise = torch.randn(b_size, nz//2, 1, 1, device=device)

            fake = netG(gaussian_noise, batch_voice)  # Generate fake image batch
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)         # Classify all fake batch with D

            errD_fake = criterion(output, label)  # D's loss on the all-fake batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            optimizerD.step()          # Update D

            "----------------------------------------------------------------------"
            "Udpate G-NET: maximize log(D(G(z)))"

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            errG = criterion(output, label)  # G's loss based on all-fake batch output of updated D

            D_G_z2 = output.mean().item()

            # Face embeddings from sphereface:
            # real_face_embeddings = sphereface(real_batch)
            # fake_face_embeddings = sphereface(fake)
            embedding_loss = cosine_criterion(real_batch, fake)
            embedding_loss.backward()
            G_loss = errG + embedding_loss

            optimizerG.step()  # Update G

            # Output training stats
            if i % 300 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, trained_epoch+opt.nepoch, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i % 500 == 0:
                with torch.no_grad():
                    fake = netG(fixed_noise, fixed_voice)
                vutils.save_image(fake.detach(), '{}/{}_fake_epoch_{}_{}.png'.format(
                                    opt.outf, opt.save_name, epoch, i), normalize=True)

        # do checkpointing
        if epoch % 3 == 0:
            print("Saving model ...")
            state_dict = {'epoch': epoch+1,
                          'generator': netG.state_dict(),
                          'discriminator': netD.state_dict()
                          }
            save_path = "saved_models/{}_ck_{}.pth.tar".format(opt.save_name, epoch)
            torch.save(state_dict, save_path)

    # Save the loss plot
    fig = plt.figure(figsize=(15, 20))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("saved_models/{}_Losses".format(opt.save_name))
