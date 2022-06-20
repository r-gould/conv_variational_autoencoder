import torch
import torch.nn as nn

from .architecture import Architecture

class MNISTArchitecture(Architecture):
    def __init__(self, latent_dim):
        super().__init__(latent_dim)

    def encoder_layers(self):
        return [
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(2304, 2 * self.latent_dim),
        ]
        
    def decoder_layers(self):
        return [
            nn.Linear(self.latent_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=2, padding=(1, 1)),
            nn.Sigmoid(),
        ]