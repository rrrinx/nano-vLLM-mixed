import os
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numeric_helpers import CASE_SEEDS, random_model_config


@pytest.fixture(scope="session", autouse=True)
def numerical_runtime():
    # Small matrix operations are faster with one CPU thread. Disable TF32 so
    # CUDA comparisons use the same float32 precision as the reference.
    threads = torch.get_num_threads()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32


@pytest.fixture
def numeric_device():
    device = torch.device(os.environ.get("MY_NANO_VLLM_TEST_DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but unavailable")
    return device


@pytest.fixture(params=CASE_SEEDS, ids=lambda seed: f"seed-{seed}")
def model_case(request, tmp_path, numeric_device):
    seed = request.param
    config, rng = random_model_config(seed, Qwen3Config)
    # Construct on CPU under a local seed; the same seed gives the same weights
    # and inputs on CPU and CUDA, without changing other tests' RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = Qwen3ForCausalLM(config).eval()
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if "norm" in name:
                    parameter.uniform_(0.5, 1.5)
    reference.save_pretrained(tmp_path, safe_serialization=True)
    tokenizer = Tokenizer(WordLevel(
        {f"t{i}_": i for i in range(config.vocab_size)}, unk_token="t0_",
    ))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0_", pad_token="t0_", eos_token="t1_",
    ).save_pretrained(tmp_path)
    reference.to(numeric_device).requires_grad_(False)
    return seed, rng, reference, tmp_path, numeric_device
