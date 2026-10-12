from ..utils.loader import load_model
from .input import prepare_decode, prepare_prefill, prepare_mixed
from ..config import Config
from ..runtime.paged_cache import PagedKVCacheManager
from .sequence import Sequence, SequenceStatus
import torch

class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        hf = config.hf_config
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        self.kv_cache_manager = PagedKVCacheManager(
            hf.num_hidden_layers, None, config.kvcache_block_size,
            hf.num_key_value_heads, head_dim,
            enable_prefix_cache=config.enable_prefix_cache,
        )
        self.model = load_model(model_dir=config.model, kv_cache_manager=self.kv_cache_manager,
                                device=config.device, dtype=config.dtype)
        self.device = next(self.model.parameters()).device
        self.max_position_embedding = self.model.max_position_embedding
        if self.device.type == "cuda" and config.num_kvcache_blocks == -1:
            self.warmup_model()
        self.allocate_kv_cache()

    def warmup_model(self):
        # Profile full prefill with the paged backend before allocating KV storage.
        config = self.config
        length = min(config.max_model_len, config.max_num_batched_tokens)
        count = min(config.max_num_seqs, config.max_num_batched_tokens // length)
        seqs = [Sequence(i, [0] * length, max_tokens=1) for i in range(count)]
        for seq in seqs:
            seq.status = SequenceStatus.RUNNING
            seq.num_scheduled_tokens = length
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.run(seqs, True)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config, hf = self.config, self.config.hf_config
        dtype = next(self.model.parameters()).dtype
        if config.num_kvcache_blocks == -1:
            if self.device.type == "cuda":
                free, total = torch.cuda.mem_get_info(self.device)
                peak = torch.cuda.max_memory_allocated(self.device)
                current = torch.cuda.memory_allocated(self.device)
                block_bytes = (2 * hf.num_hidden_layers * config.kvcache_block_size
                               * hf.num_key_value_heads * self.kv_cache_manager.head_dim * dtype.itemsize)
                config.num_kvcache_blocks = int(
                    total * config.gpu_memory_utilization - (total - free) - peak + current
                ) // block_bytes
            else:
                config.num_kvcache_blocks = (
                    (config.max_model_len + config.kvcache_block_size - 1)
                    // config.kvcache_block_size * config.max_num_seqs
                )
        assert config.num_kvcache_blocks * config.kvcache_block_size >= config.max_model_len, (
            "KV pool cannot fit max_model_len; reduce max_model_len or increase the KV budget"
        )
        # Existing attention backends already reference this manager.
        self.kv_cache_manager.reserve_storage(config.num_kvcache_blocks, self.device, dtype)

    def _check_before(self, ctx, input_ids, positions, check_cache=True):
        if check_cache:
            for seq_id, cache_len in zip(ctx.seq_ids, ctx.cache_lens):
                if self.kv_cache_manager.lengths(seq_id) != cache_len:
                    raise ValueError("KV length does not match the input plan")

        if ctx.max_seq_lens_k > self.max_position_embedding:
            raise ValueError("Input exceeds the RoPE position capacity")
        if int(positions.min()) < 0 or int(positions.max()) >= self.max_position_embedding:
            raise ValueError("Position is outside the RoPE capacity")

        vocab_size = self.model.model.embed_tokens.num_embeddings
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= vocab_size:
            raise ValueError("Token ID is outside the vocabulary")

    def _check_after(self, ctx):
        for seq_id, expected in zip(ctx.seq_ids, ctx.context_lens):
            if self.kv_cache_manager.lengths(seq_id) != expected:
                raise RuntimeError("KV length after forward does not match the input plan")

    @torch.inference_mode()
    def run_logits(self, seqs, is_prefill):
        if is_prefill is True:
            prepared = prepare_prefill(seqs, self.device)
        elif is_prefill is False:
            prepared = prepare_decode(seqs, self.device)
        else:
            prepared = prepare_mixed(seqs, self.device)


        input_ids, positions, ctx =  prepared

        check_cache = self.kv_cache_manager.k_pool is not None
        self._check_before(ctx, input_ids, positions, check_cache=check_cache)

        ctx = self.kv_cache_manager.prepare_ctx(ctx)

        logits = self.model(input_ids, positions, ctx)
        if check_cache:
            self._check_after(ctx)
        # Model 返回所有 query；Runner 每请求只返回最后位置的一行。
        index = torch.tensor(ctx.last_indices, dtype=torch.long, device=logits.device)
        return logits.index_select(0, index)

    def run(self, seqs, is_prefill):
        return [int(t) for t in self.run_logits(seqs, is_prefill).argmax(-1)]

    def run_batch(self, plan):
        plan.validate_live()
        candidates = self.run(plan.seqs, plan.legacy_flag)
        plan.validate_live()  # Forward must not advance Sequence C/Q/N.
        if len(candidates) != len(plan.entries):
            raise ValueError("one candidate per batch entry is required")
        return [token if entry.should_sample else None
            for entry, token in zip(plan.entries, candidates)]

    def register(self, seqs):
        self.kv_cache_manager.register(seqs)

    def release(self, seq_id):
        self.kv_cache_manager.release(seq_id)
