import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from architectures.mnist import MNISTArchitecture
from model.autoencoder import AutoEncoder
from utils.visualize import plot_losses, plot_latent_space, plot_manifold, plot_reconstruction

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = MNIST(root=".data", train=True, download=True, transform=transform)
    test_ds = MNIST(root=".data", train=False, download=True, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

    latent_dim = 2
    arch = MNISTArchitecture(latent_dim)

    autoencoder = AutoEncoder(arch)
    optim = torch.optim.Adam(autoencoder.parameters(), eps=1e-3)
    autoencoder.compile(optim)
    losses = autoencoder.train(train_dl, test_dl, 32)

    # Visualization

    plot_losses(losses)
    plot_latent_space(autoencoder.encoder, train_dl)
    plot_manifold(autoencoder.decoder, 16)
    plot_reconstruction(autoencoder, train_ds)

if __name__ == "__main__":
    main()