import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def IMAGENET(batch_size):
    dataset = dset.ImageNet(root="datasets", download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=8, batch_size=batch_size)
    return dataloader

if __name__ == "__main__":
    dataloader = IMAGENET(32)
    batch = next(iter(dataloader))
    print(batch[0].size())
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()