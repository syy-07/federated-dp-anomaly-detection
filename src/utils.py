from __future__ import annotations
import os, random, json, time
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
