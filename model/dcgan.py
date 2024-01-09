import torch
import torch.nn as nn

class DCDiscriminator(nn.Module):
    def __init__(self, input_dimension, channel):
        super(DCDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channel, input_dimension, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_dimension, input_dimension * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_dimension * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_dimension * 2, input_dimension * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_dimension * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_dimension * 4, input_dimension * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(input_dimension * 8, 1, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.model(data)
    
class DCGenerator(nn.Module):
    def __init__(self, input_dimension, channel, latent_size=100):
        super(DCGenerator, self).__init__()
        self.input_dimension = input_dimension
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_size, input_dimension * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(input_dimension * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(input_dimension * 8, input_dimension * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_dimension * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(input_dimension * 4, input_dimension * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_dimension * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(input_dimension * 2, input_dimension, 4, 2, 1, bias=False),
            nn.BatchNorm2d(input_dimension * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(input_dimension, channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, data):
        return self.model(data)

    def generate_noise(self, n):
        noise = torch.randn(n, self.input_dimension ** 2)
        return noise