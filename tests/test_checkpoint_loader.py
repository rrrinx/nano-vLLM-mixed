"""Single-file and HF-indexed Qwen3 checkpoints, without model downloads."""

import json

import pytest
import torch
from safetensors.torch import save_file

from nanovllm import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import postprocess_one
from nanovllm.engine.sequence import Sequence
from nanovllm.utils import loader
from numeric_helpers import allocate_decode, checked_runner_logits, hf_greedy_tokens, random_tokens
from prefix_helpers import attach_request


def write_index(directory, weight_map):
    path = directory / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return path


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_single_file_preserves_tensor_values_and_dtype(tmp_path, dtype):
    expected = {"weight": torch.arange(12, dtype=dtype).reshape(3, 4)}
    save_file(expected, tmp_path / "model.safetensors")
    # A stale index must not replace the canonical single-file checkpoint.
    write_index(tmp_path, {"unused": "missing.safetensors"})
    actual = loader.load_state_dict_from_dir(tmp_path)
    assert actual.keys() == expected.keys()
    assert actual["weight"].device.type == "cpu"
    torch.testing.assert_close(actual["weight"], expected["weight"], rtol=0, atol=0)


def test_index_loads_only_referenced_shards_once_with_original_dtypes(tmp_path, monkeypatch):
    shards = {
        "part-a.safetensors": {"a": torch.arange(6, dtype=torch.bfloat16),
                               "b": torch.ones(2, dtype=torch.float32)},
        "part-b.safetensors": {"c": torch.arange(4, dtype=torch.float16)},
        "part-c.safetensors": {"d": torch.zeros(3)},
    }
    weight_map = {}
    expected = {}
    for filename, tensors in shards.items():
        save_file(tensors, tmp_path / filename)
        weight_map.update({name: filename for name in tensors})
        expected.update(tensors)
    write_index(tmp_path, weight_map)
    # Glob-merging all safetensors would accidentally overwrite a valid weight.
    save_file({"a": torch.zeros(6)}, tmp_path / "unrelated.safetensors")
    original_load, calls = loader.load_file, []

    def observed_load(path, **kwargs):
        calls.append(path)
        return original_load(path, **kwargs)

    monkeypatch.setattr(loader, "load_file", observed_load)
    actual = loader.load_state_dict_from_dir(tmp_path)
    assert calls == [str(tmp_path / filename) for filename in sorted(shards)]
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("weight_map", [None, {}, [], {"a": None}, {"a": ""}, {"": "part.safetensors"}])
def test_invalid_index_has_clear_error(tmp_path, weight_map):
    write_index(tmp_path, weight_map)
    with pytest.raises(ValueError, match="weight_map"):
        loader.load_state_dict_from_dir(tmp_path)


@pytest.mark.parametrize("filename", ["../part.safetensors", "part.bin", "absolute"])
def test_invalid_shard_path_is_rejected(tmp_path, filename):
    if filename == "absolute":
        filename = str(tmp_path / "part.safetensors")
    write_index(tmp_path, {"a": filename})
    with pytest.raises(ValueError, match="分片文件名无效"):
        loader.load_state_dict_from_dir(tmp_path)


def test_missing_shard_is_detected_before_loading_any_tensors(tmp_path, monkeypatch):
    save_file({"a": torch.ones(1)}, tmp_path / "first.safetensors")
    write_index(tmp_path, {"a": "first.safetensors", "b": "missing.safetensors"})
    monkeypatch.setattr(loader, "load_file", lambda *a, **k: pytest.fail("preflight must fail before tensor load"))
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        loader.load_state_dict_from_dir(tmp_path)


@pytest.mark.parametrize("tensors", [{"b": torch.ones(1)}, {"a": torch.ones(1), "b": torch.zeros(1)}])
def test_shard_tensor_keys_must_match_index(tmp_path, tensors):
    save_file(tensors, tmp_path / "part.safetensors")
    write_index(tmp_path, {"a": "part.safetensors"})
    with pytest.raises(RuntimeError, match="与权重索引不一致"):
        loader.load_state_dict_from_dir(tmp_path)


def test_duplicate_tensor_in_another_shard_is_not_silently_overwritten(tmp_path):
    save_file({"a": torch.ones(1)}, tmp_path / "first.safetensors")
    save_file({"a": torch.zeros(1), "b": torch.zeros(1)}, tmp_path / "second.safetensors")
    write_index(tmp_path, {"a": "first.safetensors", "b": "second.safetensors"})
    with pytest.raises(RuntimeError, match="多余.*a"):
        loader.load_state_dict_from_dir(tmp_path)


@pytest.mark.parametrize("has_shard", [False, True])
def test_missing_single_file_and_index_explain_required_layout(tmp_path, has_shard):
    if has_shard:
        save_file({"a": torch.ones(1)}, tmp_path / "model-00001-of-00002.safetensors")
    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json") as caught:
        loader.load_state_dict_from_dir(tmp_path)
    assert str(tmp_path) in str(caught.value)


@torch.inference_mode()
def test_hf_sharded_qwen3_preserves_weights_logits_kv_and_greedy(model_case):
    seed, rng, reference, model_dir, device = model_case
    sharded = model_dir / "sharded"
    reference.save_pretrained(sharded, safe_serialization=True, max_shard_size="5KB")
    paths, weight_map = loader.resolve_checkpoint_files(sharded)
    assert weight_map is not None and len(paths) > 1, f"seed={seed}"
    single_state = loader.load_state_dict_from_dir(model_dir)
    shard_state = loader.load_state_dict_from_dir(sharded)
    assert single_state.keys() == shard_state.keys()
    for name in single_state:
        torch.testing.assert_close(shard_state[name], single_state[name], rtol=0, atol=0)

    runner = ModelRunner(Config(
        str(sharded), device=str(device), dtype="float32", max_model_len=32,
        max_num_batched_tokens=32, max_num_seqs=1,
        kvcache_block_size=4, num_kvcache_blocks=16,
    ))
    for name, parameter in runner.model.named_parameters():
        torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=0, atol=0)
    prompt = random_tokens(rng, 7, reference.config.vocab_size)
    seq = Sequence(17, prompt, 3)
    manager = runner.kv_cache_manager
    try:
        assert attach_request(manager, seq) == 0
        for index in range(3):
            if index:
                seq.num_scheduled_tokens = 1
                allocate_decode(manager, [seq])
            # Compare every query logit and each layer's KV against HF's full
            # history, then check greedy tokens across a cache block boundary.
            logits = checked_runner_logits(runner, [seq], index == 0, reference)
            manager.publish_seq(seq)
            postprocess_one(seq, int(logits.argmax(-1)[0]))
        assert seq.completion_token_ids == hf_greedy_tokens(reference, prompt, 3)
    finally:
        if seq.seq_id in manager.layer_lengths:
            manager.release(seq.seq_id)
