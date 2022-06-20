import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        self.network = nn.Sequential(*layers)

    def forward(self, code):
        return self.network(code)