import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from time import perf_counter

from transformers import AutoTokenizer

from nanovllm import Config, LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Run local Qwen3 greedy generation.")
    parser.add_argument("model", nargs="?", default=Config.model_path)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=Config.dtype)
    parser.add_argument("--device", default=Config.device)
    args = parser.parse_args()
    path = args.model
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    llm = LLM(path, dtype=args.dtype, device=args.device)
    parameter = next(llm.model_runner.model.parameters())
    print(f"Runtime: device={parameter.device}, model_dtype={parameter.dtype}, "
          f"kv_dtype={llm.model_runner.kv_cache_manager.k_pool.dtype}", flush=True)
    sampling_params = SamplingParams(max_tokens=4)
    prompts = [
        "count all the prime number between 1 and 100.",
        "count all the prime number between 1 and 100.",
        "count all the prime number between 1 and 100.",
        "count all the prime number between 1 and 100.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    generate_start = perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    generate_elapsed = perf_counter() - generate_start
    print(f"Generate total time: {generate_elapsed:.3f} s")
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}\n")


if __name__ == "__main__":
    main()
