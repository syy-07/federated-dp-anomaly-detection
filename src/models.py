from __future__ import annotations
import torch, torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[64,32], latent_dim=12):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d,h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(d, latent_dim)

    def forward(self, x):
        h = self.net(x)
        z = self.mu(h)
        return z

class MLPDecoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[32,64], latent_dim=12):
        super().__init__()
        layers = []
        d = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(d,h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, input_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class AE(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[64,32], latent_dim=12):
        super().__init__()
        self.enc = MLPEncoder(input_dim, hidden_dims, latent_dim)
        self.dec = MLPDecoder(input_dim, hidden_dims[::-1], latent_dim)

    def forward(self, x):
        z = self.enc(x)
        xrec = self.dec(z)
        return xrec
