import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_losses(losses):
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()

def plot_latent_space(encoder, train_dl, batches=float("inf")):
    with torch.no_grad():
        for count, (images, target) in enumerate(train_dl):
            if count >= batches:
                break
            encoded = encoder(images)
            code, _ = encoder.reparameterize(encoded)
            
            plt.scatter(code[:, 0], code[:, 1], c=target, cmap="viridis")
    
    plt.colorbar()
    plt.show()

def plot_manifold(decoder, num=16):
    x_vals = np.linspace(-3, 3, num)
    y_vals = np.linspace(-3, 3, num)

    canvas = np.empty((28*num, 28*num))
    with torch.no_grad():
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                code = torch.Tensor([[x, y]])
                decoded = decoder(code)
                canvas[i*28:(i+1)*28, j*28:(j+1)*28] = decoded.view(28, 28)

    plt.figure(figsize=(8, 10))
    plt.imshow(canvas, cmap="gray")
    plt.show()

def plot_reconstruction(autoencoder, train_ds, count=10):
    canvas = np.empty((28*count, 28*2))
    with torch.no_grad():
        for i, (image, _) in enumerate(train_ds):
            if i >= count:
                break
            recon, _ = autoencoder(image.view(1, 1, 28, 28))
            canvas[i*28:(i+1)*28, :28] = image.view(28, 28)
            canvas[i*28:(i+1)*28, 28:] = recon.view(28, 28)

    plt.figure(figsize=(8, 10))
    plt.imshow(canvas, cmap="gray")
    plt.show()