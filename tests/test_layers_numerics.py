import random

import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.mlp import SwiGLU
from nanovllm.layers.norm import RMSNorm
from nanovllm.layers.rope import RotaryEmbedding
from numeric_helpers import CASE_SEEDS, TENSOR_SEEDS, random_tensor


@pytest.mark.parametrize("seed", TENSOR_SEEDS, ids=lambda seed: f"seed-{seed}")
@torch.inference_mode()
def test_random_layers_against_float64(seed, numeric_device):
    rng = random.Random(seed)
    hidden = rng.choice([4, 7, 16, 32, 63])
    eps = rng.choice([1e-6, 1e-5, 1e-3])
    x = random_tensor(rng, (rng.randint(1, 4), rng.randint(1, 9), hidden),
                      numeric_device, scale=10 ** rng.uniform(-4, 3))
    norm = RMSNorm(hidden, eps).to(numeric_device)
    norm.weight.copy_(random_tensor(rng, (hidden,), numeric_device))
    expected_norm = x.double() * (x.double().square().mean(-1, keepdim=True) + eps).rsqrt()
    expected_norm *= norm.weight.double()
    torch.testing.assert_close(norm(x), expected_norm.float(), atol=2e-6, rtol=2e-5)

    dim = rng.choice([2, 4, 8, 12, 32, 64])
    max_position = rng.choice([16, 32, 64, 96])
    base = rng.choice([100.0, 10_000.0, 1_000_000.0])
    positions = torch.tensor(
        [0, max_position - 1] + [rng.randrange(max_position) for _ in range(rng.randint(1, 15))],
        device=numeric_device,
    )
    query = random_tensor(rng, (len(positions), rng.randint(1, 8), dim), numeric_device)
    key = random_tensor(rng, (len(positions), rng.randint(1, 4), dim), numeric_device)
    rope = RotaryEmbedding(dim, max_position, base).to(numeric_device)
    assert rope.cos_sin_cache.device == torch.empty(0, device=numeric_device).device
    assert rope.cos_sin_cache.dtype == torch.float32
    rotated = rope(positions, query, key)
    frequencies = base ** (-torch.arange(0, dim, 2, device=numeric_device, dtype=torch.float64) / dim)
    angles = positions.double()[:, None, None] * frequencies[None, None]
    for actual, source in zip(rotated, (query, key)):
        first, second = source.double().chunk(2, -1)
        expected = torch.cat((first * angles.cos() - second * angles.sin(),
                              second * angles.cos() + first * angles.sin()), -1)
        torch.testing.assert_close(actual, expected.float(), atol=2e-5, rtol=2e-5)

    intermediate = rng.choice([8, 17, 32, 65])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        mlp = SwiGLU(hidden, intermediate).to(numeric_device)
    for parameter in mlp.parameters():
        parameter.copy_(random_tensor(rng, parameter.shape, numeric_device, scale=0.1))
    x = random_tensor(rng, (rng.randint(1, 12), hidden), numeric_device,
                      scale=rng.choice([0.01, 0.1, 1.0, 3.0]))
    gate = F.linear(x.double(), mlp.gate_proj.weight.double())
    up = F.linear(x.double(), mlp.up_proj.weight.double())
    expected = F.linear(gate * gate.sigmoid() * up, mlp.down_proj.weight.double())
    torch.testing.assert_close(mlp(x), expected.float(), atol=3e-5, rtol=2e-5)


@pytest.mark.parametrize("seed", CASE_SEEDS, ids=lambda seed: f"seed-{seed}")
@torch.inference_mode()
def test_random_rope_device_and_dtype_moves_rebuild_float32_table(seed, numeric_device, monkeypatch):
    rng = random.Random(seed)
    dim = rng.choice([2, 4, 8, 12, 32, 64, 128])
    length = rng.choice([16, 32, 64, 128, 256])
    base = rng.choice([100.0, 10_000.0, 1_000_000.0])
    rope = RotaryEmbedding(dim, length, base)
    builds = []
    original = rope._build_cache

    def observed_build(device):
        builds.append(torch.device(device))
        return original(device)

    monkeypatch.setattr(rope, "_build_cache", observed_build)
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    rng.shuffle(dtypes)
    canonical_inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # CPU -> target -> CPU -> target also exercises a populated CPU table
    # crossing into CUDA, rather than merely checking the final device label.
    targets = [numeric_device, torch.device("cpu"), numeric_device]
    for device, dtype in zip(targets, dtypes):
        rope.to(device=device, dtype=dtype)
        actual_device = torch.empty(0, device=device).device
        assert builds[-1] == actual_device
        assert rope.cos_sin_cache.device == actual_device
        assert rope.cos_sin_cache.dtype == torch.float32
        positions = torch.arange(length, dtype=torch.float32, device=device)
        angles = positions[:, None] * canonical_inv.to(device)[None]
        expected = torch.cat((angles.cos(), angles.sin()), -1).unsqueeze(1)
        torch.testing.assert_close(rope.cos_sin_cache, expected, atol=0, rtol=0)
        query = random_tensor(rng, (3, 2, dim), device).to(dtype)
        key = random_tensor(rng, (3, 1, dim), device).to(dtype)
        indices = torch.tensor([0, length - 1, rng.randrange(length)], device=device)
        rotated = rope(indices, query, key)
        for output, source in zip(rotated, (query, key)):
            first, second = source.float().chunk(2, -1)
            cos, sin = expected[indices].chunk(2, -1)
            wanted = torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(dtype)
            torch.testing.assert_close(output, wanted, atol=0, rtol=0)
        assert "cos_sin_cache" not in rope.state_dict()
    assert len(builds) == 3
