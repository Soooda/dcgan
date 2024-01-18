import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

data_root = 'datasets/celeba'
dataset_folder = f'{data_root}/img_align_celeba'
image_size = 64
num_workers = 8
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
pin_memory = True if device.type == 'cuda' else False

class CelebADataset(Dataset):
  def __init__(self, root, transform=transform):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root)

    self.root_dir = root
    self.transform = transform 
    self.image_names = sorted(image_names)

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img

def CELEBA(batch_size):
    dataset = CelebADataset(root=dataset_folder, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    return dataloader

if __name__ == "__main__":
    dataloader = CELEBA(32)
    batch = next(iter(dataloader))
    print(batch.size())

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    # for i in range(batch[0].size()[0]):
    #     img = batch[0][i].view(image_size, image_size)
    #     plt.figure(figsize=(4, 4))
    #     plt.axis('Off')
    #     plt.imshow(img)
    #     plt.show()