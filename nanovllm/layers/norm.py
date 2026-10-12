import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        # shape: x = [B, T, H]
        original_dtype = x.dtype
        x = x.float()
        v = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        return x.to(original_dtype) * (self.weight)
