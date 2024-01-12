import torch
import torch.nn as nn
import torch.optim as optim
import os

from model.dcgan import Generator, Discriminator, weights_init
from data.mnist import MNIST

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

num_epochs = 100
batch_size = 128
image_size = 28
num_channels = 1
latent_feature = 100
learning_rate = 2e-4
beta1 = 0.5
num_gpu = 1

dataloader = MNIST(batch_size=batch_size)
netG = Generator(num_gpu, latent_feature, num_channels).to(device)
netD = Discriminator(num_gpu, num_channels).to(device)
if (device.type == 'cuda') and (num_gpu > 1):
    netG = nn.DataParallel(netG, list(range(num_gpu)))
    netD = nn.DataParallel(netD, list(range(num_gpu)))
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batch_size, latent_feature, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

for epoch in range(1, num_epochs + 1):
    checkpoint = os.sep.join(("checkpoints", str(epoch) + ".pth"))
    if os.path.exists(checkpoint):
        if os.path.exists(os.sep.join(("checkpoints", str(epoch + 1) + ".pth"))):
            continue
        temp = torch.load(checkpoint)
        netD.load_state_dict(temp["netD"])
        netG.load_state_dict(temp["netG"])
        optimizerD.load_state_dict(temp["optimizerD"])
        optimizerG.load_state_dict(temp["optimizerG"])
        continue

    for i, data in enumerate(dataloader, start=0):
        optimizerD.zero_grad()
        
        # Train with real data
        imgs = data[0].to(device)
        labels = torch.full((imgs.size(0),), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(imgs).view(-1)
        # Calculate the loss on all-real batch
        error_real = criterion(output, labels)
        error_real.backward()
        D_x = output.mean().item()

        # Train with fake data
        noise = torch.randn(imgs.size(0), latent_feature, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        labels.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the fake batch
        error_fake = criterion(output, labels)
        error_fake.backward()
        D_G_z1 = output.mean().item()
        # Computer error of D as sum over the fake and real batches
        errorD = error_real + error_fake
        optimizerD.step()

        optimizerG.zero_grad()
        labels.fill_(real_label)
        output = netD(fake).view(-1)
        errorG = criterion(output, labels)
        errorG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

    print("Epoch {:<3} Loss_D: {:<.4f} Loss_G: {:<.4f} D(x): {:<.4f} D(G(z)): {:<.4f} / {:<.4f}".format(epoch, errorD.item(), errorG.item(), D_x, D_G_z1, D_G_z2))
    checkpoints = {
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "optimizerD": optimizerD.state_dict(),
    }

    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    torch.save(checkpoints, os.sep.join(("checkpoints", str(epoch) + ".pth")))