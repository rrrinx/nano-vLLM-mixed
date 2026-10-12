import torch

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from numeric_helpers import (
    advance_with_random_tokens, checked_runner_logits,
    make_sequences, shuffled_blocks, allocate_registered_prefill, allocate_decode,
)


@torch.inference_mode()
def test_full_prefill_before_and_after_pool_allocation(model_case, monkeypatch):
    _, rng, reference, model_dir, device = model_case
    hf = reference.config
    block_size = rng.choice([1, 2, 3, 4, 7, 8])
    decode_steps = rng.randint(3, 7)
    seqs = make_sequences(rng, hf.vocab_size, block_size,
                          max_length=hf.max_position_embeddings - decode_steps)
    config = Config(
        str(model_dir), device=str(device), dtype="float32",
        max_model_len=hf.max_position_embeddings,
        max_num_batched_tokens=hf.max_position_embeddings * len(seqs),
        max_num_seqs=len(seqs), kvcache_block_size=block_size,
        num_kvcache_blocks=len(seqs) * ((hf.max_position_embeddings + block_size - 1) // block_size),
    )
    # Pause construction before allocation to exercise the real runner's
    # no-cache warmup forward. Neither model math nor logits are mocked.
    with monkeypatch.context() as deferred:
        deferred.setattr(ModelRunner, "allocate_kv_cache", lambda self: None)
        runner = ModelRunner(config)
    manager = runner.kv_cache_manager
    rng.shuffle(seqs)
    uncached = checked_runner_logits(runner, seqs, True, reference)

    runner.allocate_kv_cache()
    manager.k_pool.fill_(torch.nan)
    manager.v_pool.fill_(torch.nan)
    shuffled_blocks(manager, rng)
    manager.register(seqs)
    allocate_registered_prefill(manager, seqs)
    cached = checked_runner_logits(runner, seqs, True, reference)
    torch.testing.assert_close(cached, uncached, atol=1e-5, rtol=1e-4)

    # The backends constructed before allocation must also produce correct
    # cached decoding once storage has been attached to the same manager.
    for _ in range(decode_steps):
        advance_with_random_tokens(seqs, rng, hf.vocab_size)
        rng.shuffle(seqs)
        allocate_decode(manager, seqs)
        checked_runner_logits(runner, seqs, False, reference)


@torch.inference_mode()
def test_runner_random_histories_batch_orders_and_recomputation(model_case):
    _, rng, reference, model_dir, device = model_case
    hf = reference.config
    block_size = rng.choice([1, 2, 3, 4, 7, 8])
    seqs = make_sequences(rng, hf.vocab_size, block_size,
                          max_length=hf.max_position_embeddings - 12)
    runner = ModelRunner(Config(
        str(model_dir), device=str(device), dtype="float32",
        max_model_len=hf.max_position_embeddings,
        max_num_batched_tokens=hf.max_position_embeddings * len(seqs),
        max_num_seqs=len(seqs), kvcache_block_size=block_size,
        num_kvcache_blocks=len(seqs) * ((hf.max_position_embeddings + block_size - 1) // block_size),
    ))
    runner.kv_cache_manager.k_pool.fill_(torch.nan)
    runner.kv_cache_manager.v_pool.fill_(torch.nan)
    shuffled_blocks(runner.kv_cache_manager, rng)
    runner.register(seqs)
    allocate_registered_prefill(runner.kv_cache_manager, seqs)
    rng.shuffle(seqs)
    checked_runner_logits(runner, seqs, True, reference)
    advance_with_random_tokens(seqs, rng, hf.vocab_size)

    for step in range(rng.randint(5, 9)):
        if step == 2:
            # Recompute a random complete history while other requests retain
            # their existing KV values and pending decode token.
            recomputed = rng.sample(seqs, rng.randint(1, len(seqs)))
            for seq in recomputed:
                runner.release(seq.seq_id)
                seq.num_cached_tokens = 0
                seq.num_scheduled_tokens = seq.num_tokens
            runner.register(recomputed)
            shuffled_blocks(runner.kv_cache_manager, rng)
            allocate_registered_prefill(runner.kv_cache_manager, recomputed)
            checked_runner_logits(runner, recomputed, True, reference)
            advance_with_random_tokens(recomputed, rng, hf.vocab_size)
        batch = rng.sample(seqs, rng.randint(1, len(seqs)))
        allocate_decode(runner.kv_cache_manager, batch)
        checked_runner_logits(runner, batch, False, reference)
        advance_with_random_tokens(batch, rng, hf.vocab_size)
