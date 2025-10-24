from __future__ import annotations
import math, torch

def clip_gradients_per_sample(grads, max_norm: float):
    # grads: tensor [B, ...] shaped list; here we accept a list of parameter per-sample grads
    total = 0.0
    for g in grads:
        total += (g.view(g.size(0), -1).pow(2).sum(dim=1))
    norms = total.sqrt()  # [B]
    scale = (max_norm / (norms + 1e-12)).clamp(max=1.0)  # [B]
    # scale each sample's grad
    scaled = []
    for g in grads:
        scaled.append(g * scale.view(-1, *([1]*(g.dim()-1))))
    return scaled, norms.mean().item()

def add_gaussian_noise(param_grads, noise_multiplier: float, max_norm: float, batch_size: int, device):
    std = noise_multiplier * max_norm
    for g in param_grads:
        noise = torch.randn_like(g[0], device=device) * std  # same shape as aggregated grad
        g[0].add_(noise / batch_size)

def rdp_gaussian(q, noise_multiplier, steps, orders=(1.25,1.5,2,3,4,8,16,32)):
    # rough RDP accountant (q: sampling rate)
    eps = []
    for a in orders:
        alpha = a
        eps_a = steps * (alpha * q*q) / (2 * (noise_multiplier**2))
        eps.append((alpha, eps_a))
    return eps
