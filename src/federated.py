from __future__ import annotations
import random, torch, numpy as np
from torch.utils.data import TensorDataset, DataLoader

def make_client_loaders(client_data, batch_size=64):
    loaders = []
    for X, y in client_data:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False))
    return loaders

def fedavg(weights):
    # weights: list of state_dicts; returns averaged state_dict
    avg = {}
    for k in weights[0].keys():
        avg[k] = sum(w[k] for w in weights) / float(len(weights))
    return avg

def select_clients(n_clients, participation_rate):
    k = max(1, int(round(n_clients * participation_rate)))
    return sorted(random.sample(range(n_clients), k))
