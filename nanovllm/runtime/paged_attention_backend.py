import torch

class PagedAttentionBackend:
    def __init__(self, paged_kv_cache_manager, layer_idx):
        self.layer_idx = layer_idx
        self.paged_kv_cache_manager = paged_kv_cache_manager

    def attend(self, q, k, v, positions, ctx):
        use_cache = ctx.block_tables is not None
        if not use_cache:
            if (self.paged_kv_cache_manager.k_pool is not None
                    or ctx.is_prefill is not True or any(ctx.cache_lens)
                    or ctx.cu_seq_lens_q != ctx.cu_seq_lens_k
                    or ctx.slot_mapping is not None):
                raise ValueError("No-cache attention requires full prefill before KV pool allocation")
        out = torch.empty_like(q)

        for i, seq_id in enumerate(ctx.seq_ids):
            lo, hi = ctx.cu_seq_lens_q[i:i + 2]

            positions_i = positions[lo:hi]


            q_i = q[lo:hi] # T, H, D
            k_i = k[lo:hi]
            v_i = v[lo:hi]

            if use_cache:
                self.paged_kv_cache_manager.append(
                    layer_idx=self.layer_idx, seq_id=seq_id,
                    k_new=k_i, v_new=v_i,
                    expected_old=ctx.cache_lens[i],
                    slots=ctx.slot_mapping[lo:hi],
                )

                k_i, v_i = self.paged_kv_cache_manager.gather(
                    layer_idx=self.layer_idx, seq_id=seq_id,
                    valid_length=ctx.context_lens[i],
                    table_snapshot=ctx.block_tables[i],
                )
            # Before pool allocation, full prefill attends to this forward's K/V.

            assert k_i.shape[0] == ctx.context_lens[i]

            # H, T, D
            q_i, k_i, v_i = q_i.transpose(0, 1), k_i.transpose(0, 1), v_i.transpose(0, 1)

            groups = q_i.shape[0] // k_i.shape[0]
            assert q_i.shape[0] % k_i.shape[0] == 0
            k_for_attn = k_i.repeat_interleave(groups, dim=0)
            v_for_attn = v_i.repeat_interleave(groups, dim=0)
            scores = (q_i @ k_for_attn.transpose(-1, -2)) * (q_i.shape[-1] ** -0.5) # [Hq,Tq,Tk]

            key_positions = torch.arange(k_i.shape[-2], device=q.device)
            allowed = key_positions[None, :] <= positions_i[:, None]
            scores = scores.masked_fill(~allowed[None], torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out[lo:hi] = (probs @ v_for_attn).transpose(0, 1)

        return out
