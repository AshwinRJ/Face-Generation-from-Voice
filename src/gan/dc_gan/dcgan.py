import argparse
import os
import random
import torch
from tqdm import tnrange
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

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--bs', type=int, default=128, help='input batch size')
parser.add_argument('--imsize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nepoch', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--load', default='', help="path to saved models")
parser.add_argument('--outf', default='saved_models/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()
print(opt)

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

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
trained_epoch = 0
if opt.load != '':
    checkpoint = torch.load(opt.load)
    trained_epoch = checkpoint['epoch']
    netD.load_state_dict(checkpoint['discriminator'])
    netG.generator.load_state_dict(checkpoint['generator'])


# Initialize BCELoss function
criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.bs, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

#trained_epoch = 1
"==========================================================================="
# Training
for epoch in tnrange(trained_epoch+1, trained_epoch+opt.nepoch+1):
    for i, batch in enumerate(train_loader, 0):
        batch_voice, face_real, gold_labels = batch
        batch_voice, gold_labels = batch_voice.to(device), gold_labels.to(device)

        "Udpate D-Net: maximize log(D(x)) + log(1 - D(G(z)))"

        "-----------Train with ALL-REAL batch-----------------"
        netD.zero_grad()

        real_cpu = face_real.to(device)
        b_size = real_cpu.size(0)

        label = torch.full((b_size,), real_label, device=device)

        # Downsample the input image
        # down_real_cpu = nn.AvgPool2d(2, stride=2)(real_cpu)

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

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.nepoch, i, len(train_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        # G_losses.append(errG.item())
        # D_losses.append(errD.item())

        if i % 100 == 0:
            vutils.save_image(real_cpu,
                              '%s/real_samples.png' % opt.outf,
                              normalize=True)
            fake = netG(fixed_noise)
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
