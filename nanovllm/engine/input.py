import torch
from .context import Context
from .sequence import Sequence, SequenceStatus

def prepare_inputs(seqs, is_prefill, device):
    if not seqs or len({s.seq_id for s in seqs}) != len(seqs):
        raise ValueError("id can not be None or the same")

    ids, positions = [], []
    cu_q, cu_k = [0], [0]
    old_lens, key_lens, query_lens = [], [], []

    for seq in seqs:
        c, q, n = seq.num_cached_tokens, seq.num_scheduled_tokens, seq.num_tokens

        if not (0 <= c and q > 0 and c + q <= n):
            raise ValueError("input Error")

        
        if seq.status not in (SequenceStatus.WAITING, SequenceStatus.RUNNING):
            raise ValueError("input requires a live request")
        if is_prefill is False:
            if seq.status != SequenceStatus.RUNNING or not (q == 1 and c == n - 1):
                raise ValueError("decode must process the uncached last token")

        

        ids.extend(seq.token_ids[c: c + q])
        positions.extend(range(c, c + q))

        query_lens.append(q)
        old_lens.append(c)
        key_lens.append(c + q)

        cu_q.append(cu_q[-1] + q)
        cu_k.append(cu_k[-1] + c + q)


    ctx = Context(
        is_prefill=is_prefill,
        seq_ids=tuple(seq.seq_id for seq in seqs),
        cu_seq_lens_q=tuple(cu_q), cu_seq_lens_k=tuple(cu_k),
        cache_lens=tuple(old_lens), context_lens=tuple(key_lens),
        max_seq_lens_q=max(query_lens), max_seq_lens_k=max(key_lens)
        )

    return (torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(positions, dtype=torch.long, device=device),
            ctx)

def prepare_prefill(seqs, device):
    return prepare_inputs(seqs, True, device)

def prepare_decode(seqs, device):
    return prepare_inputs(seqs, False, device)

def prepare_mixed(seqs, device):
    return prepare_inputs(seqs, None, device)