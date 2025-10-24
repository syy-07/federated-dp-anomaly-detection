import torch
from src.models import AE

def test_forward():
    m = AE(16, [32,16], 8)
    x = torch.randn(5,16)
    y = m(x)
    assert y.shape == x.shape
