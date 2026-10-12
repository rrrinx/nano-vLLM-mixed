import atexit

import torch

from nanovllm import LLM, SamplingParams
from numeric_helpers import (
    checked_runner_logits, hf_greedy_tokens, random_tokens, shuffled_blocks,
)


def attach_numerical_oracle(engine, reference, monkeypatch, recomputed=None):
    runner = engine.model_runner
    original = runner.run_logits

    def checked(seqs, is_prefill):
        if recomputed is not None:
            recomputed.extend(seq.seq_id for seq in seqs
                              if is_prefill and seq.num_completion_tokens > 0)
        return checked_runner_logits(runner, seqs, is_prefill, reference, original)

    monkeypatch.setattr(runner, "run_logits", checked)


@torch.inference_mode()
def test_public_greedy_tokens_and_logits_against_hf(model_case, monkeypatch):
    _, rng, reference, model_dir, device = model_case
    hf = reference.config
    block_size = rng.choice([1, 2, 3, 4, 7, 8])
    count = rng.randint(2, 5)
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32",
        max_model_len=hf.max_position_embeddings,
        max_num_batched_tokens=hf.max_position_embeddings * count,
        max_num_seqs=rng.randint(1, count), kvcache_block_size=block_size,
        num_kvcache_blocks=count * ((hf.max_position_embeddings + block_size - 1) // block_size),
    )
    try:
        attach_numerical_oracle(engine, reference, monkeypatch)
        for wave in range(2):
            prompts = [random_tokens(rng, rng.randint(1, 12), hf.vocab_size) for _ in range(count)]
            budgets = [rng.randint(1, 7) for _ in prompts]
            if wave == 0:
                budgets[0] = 0
            # Choose an EOS actually produced by this random model to exercise
            # early stopping numerically; some requests deliberately ignore it.
            eos = hf_greedy_tokens(reference, prompts[-1], 1)[0]
            params = [SamplingParams(max_tokens=budget, ignore_eos=rng.choice([False, True]))
                      for budget in budgets]
            params[-1].ignore_eos = False
            expected = [hf_greedy_tokens(reference, prompt, budget,
                                         None if param.ignore_eos else eos)
                        for prompt, budget, param in zip(prompts, budgets, params)]
            # Text and token-ID inputs must reach the same numerical model path.
            inputs = [" ".join(f"t{token}_" for token in prompt) if rng.choice([False, True]) else prompt
                      for prompt in prompts]
            shuffled_blocks(engine.model_runner.kv_cache_manager, rng)
            actual = engine.generate(inputs, params, use_tqdm=False, eos_token_id=eos)
            assert [result["token_ids"] for result in actual] == expected
    finally:
        engine.exit()
        atexit.unregister(engine.exit)


@torch.inference_mode()
def test_continuous_batch_logits_and_tokens_under_preemption(model_case, monkeypatch):
    _, rng, reference, model_dir, device = model_case
    hf = reference.config
    block_size = rng.choice([2, 3, 4, 7])
    # Two full pages admit both initial prompts, then force eviction when the
    # first decode needs a third page. All individual requests still fit.
    context_length = 2 * block_size
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32",
        max_model_len=context_length,
        max_num_batched_tokens=context_length * rng.randint(1, 3),
        max_num_seqs=rng.randint(2, 4),
        kvcache_block_size=block_size, num_kvcache_blocks=2,
    )
    arrivals = [(0, random_tokens(rng, block_size, hf.vocab_size), rng.randint(2, block_size + 1))
                for _ in range(2)]
    for _ in range(rng.randint(2, 5)):
        length = rng.randint(1, block_size)
        arrivals.append((rng.randint(2, 6), random_tokens(rng, length, hf.vocab_size),
                         rng.randint(1, context_length - length + 1)))
    arrivals.sort(key=lambda item: item[0])
    recomputed = []
    expected = {}
    actual = {}
    try:
        attach_numerical_oracle(engine, reference, monkeypatch, recomputed)
        shuffled_blocks(engine.model_runner.kv_cache_manager, rng)
        limit = sum(budget for _, _, budget in arrivals) + 20
        for tick in range(limit):
            for arrival, prompt, budget in arrivals:
                if arrival == tick:
                    sid = engine.add_request(prompt, SamplingParams(max_tokens=budget, ignore_eos=True))
                    expected[sid] = hf_greedy_tokens(reference, prompt, budget)
            if not engine.is_finished():
                actual.update(engine.step())
            if tick >= arrivals[-1][0] and engine.is_finished():
                break
        # Guard that numerical comparisons actually traversed recomputed KV,
        # then compare final greedy token IDs with independent serial forwards.
        assert recomputed, "case did not exercise recomputed numerical outputs"
        assert actual == expected
    finally:
        engine.exit()
        atexit.unregister(engine.exit)
