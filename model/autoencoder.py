import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .encoder import Encoder
from .decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self, arch):
        super().__init__()

        self.encoder = Encoder(arch.encoder_layers(), arch.latent_dim)
        self.decoder = Decoder(arch.decoder_layers())
        self.compiled = False

    def forward(self, images):
        encoded = self.encoder(images)
        code, (mean, log_std) = self.encoder.reparameterize(encoded)
        decoded = self.decoder(code)
        return decoded, (mean, log_std)

    def compile(self, optim):
        self.optim = optim
        self.compiled = True

    def train(self, train_dl, test_dl, epochs):
        if not self.compiled:
            raise RunTimeError("Must run compile method before training.")

        losses = []
        epoch_loss = 0
        for epoch in range(1, epochs+1):
            print("Epoch:", epoch)
            for images, _ in train_dl:
                decoded, (mean, log_std) = self.forward(images)
                loss = self.loss(images, decoded, mean, log_std)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        
            epoch_loss = self.test(test_dl)
            losses.append(epoch_loss)
        return losses

    def test(self, test_dl):
        total_loss = 0
        with torch.no_grad():
            for images, _ in test_dl:
                decoded, (mean, log_std) = self.forward(images)
                loss = self.loss(images, decoded, mean, log_std)
                total_loss += loss
        total_loss /= len(test_dl.dataset) / test_dl.batch_size
        return total_loss
            

    def loss(self, images, decoded, mean, log_std):
        # Assuming L = 1, Gaussian
        KL_loss = -0.5 * torch.sum(1 + 2*log_std - torch.square(mean) - torch.exp(2*log_std))
        reconstruction_loss = torch.sum(torch.square(images - decoded))
        loss = (KL_loss + reconstruction_loss) / len(images)
        return loss