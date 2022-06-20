import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, layers, latent_dim):
        super().__init__()
        
        self.network = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, images):
        return self.network(images)

    def reparameterize(self, encoded):
        mean, log_std = torch.split(encoded, self.latent_dim, dim=1)
        noise = torch.normal(torch.zeros(mean.shape), torch.ones(log_std.shape))
        code = mean + noise * torch.exp(log_std)
        return code, (mean, log_std)