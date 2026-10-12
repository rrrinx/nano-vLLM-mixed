import torch
from torch import nn

def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim:int,
        max_position_embeddings: int,
        base: float,
        ):
        super().__init__()
        self.head_dim = head_dim
        assert head_dim % 2 == 0
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("cos_sin_cache", self._build_cache(torch.device("cpu")), persistent=False)

    def _build_cache(self, device):
        # Keep frequencies canonical across devices, but calculate trig on the
        # target device in float32 even when model weights use a lower dtype.
        inv = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        angles = torch.arange(self.max_position_embeddings, dtype=torch.float32, device=device).unsqueeze(-1) * inv.to(device)
        return torch.cat((angles.cos(), angles.sin()), dim=-1).unsqueeze(1)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.cos_sin_cache = self._build_cache(self.cos_sin_cache.device)
        return self

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions] # [B, T, 1, D]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key
