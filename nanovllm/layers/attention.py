import torch
from torch import nn
from .norm import RMSNorm
from .rope import RotaryEmbedding

class Attention(nn.Module):

    def __init__(
        self,
        hidden_size,
        heads_num,
        heads_kv_num,
        heads_dim,
        max_position_embeddings,
        base,
        rms_norm_eps,
        layer_idx,
        kv_cache_manager,
    ):
        super().__init__()
        self.heads_num = heads_num
        self.heads_dim = heads_dim
        self.heads_kv_num = heads_kv_num

        self.q_proj = nn.Linear(hidden_size, heads_dim * heads_num, False)
        self.k_proj = nn.Linear(hidden_size, heads_dim * heads_kv_num, False)
        self.v_proj = nn.Linear(hidden_size, heads_dim * heads_kv_num, False)
        self.o_proj = nn.Linear(heads_dim * heads_num, hidden_size, False)

        # HF 的 Qwen3Attention 对 q_norm/k_norm 用的是 config.rms_norm_eps，不是各自的默认值。
        self.q_norm = RMSNorm(heads_dim, rms_norm_eps)
        self.k_norm = RMSNorm(heads_dim, rms_norm_eps)

        self.rope = RotaryEmbedding(head_dim=heads_dim,
                                    max_position_embeddings=max_position_embeddings,
                                    base=base)
        self.backend = kv_cache_manager.make_backend(layer_idx)

    def forward(self, x, positions, ctx):
        # x: [Nq,H]，positions: [Nq]，每条请求使用自己的绝对位置。

        nq = x.shape[0]

        q = self.q_proj(x).reshape(nq, self.heads_num, self.heads_dim)    # [Nq,Hq,D]
        k = self.k_proj(x).reshape(nq, self.heads_kv_num, self.heads_dim) # [Nq,Hkv,D]
        v = self.v_proj(x).reshape(nq, self.heads_kv_num, self.heads_dim) # [Nq,Hkv,D]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rope(positions, q, k)

        out = self.backend.attend(q, k, v, positions, ctx)

        return self.o_proj(out.reshape(nq, -1))
