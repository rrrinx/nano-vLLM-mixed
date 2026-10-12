import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.benchmark import digest, write_json


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_trace(trace, *, vocab_size=None, max_model_len=None,
                   block_size=None, num_blocks=None):
    if not isinstance(trace, dict):
        raise ValueError("trace must be an object")
    if trace.get("schema") != "online-v1":
        raise ValueError("expected online-v1")
    requests = trace.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a nonempty list")
    seen, last = set(), -1.0
    for r in requests:
        required = {"id", "arrival_s", "group", "dataset_index", "prompt_token_ids", "max_tokens"}
        if not isinstance(r, dict) or not required <= r.keys():
            raise ValueError("request is missing required trace fields")
        rid, arrival = r["id"], r["arrival_s"]
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("request IDs must be nonempty and unique")
        if (type(arrival) not in (int, float) or not math.isfinite(arrival)
                or arrival < 0 or arrival < last):
            raise ValueError("arrivals must be finite, nonnegative and sorted")
        if not isinstance(r["group"], str) or not r["group"] or r["group"] == "all":
            raise ValueError("group must be a nonempty name other than reserved 'all'")
        if type(r["dataset_index"]) is not int:
            raise ValueError("dataset_index must be int")
        ids, cap = r["prompt_token_ids"], r["max_tokens"]
        if (not isinstance(ids, list) or not ids
                or any(type(t) is not int or t < 0 for t in ids)):
            raise ValueError("invalid token IDs")
        if vocab_size is not None and any(t >= vocab_size for t in ids):
            raise ValueError("token ID outside model vocabulary")
        if type(cap) is not int or cap < 1:
            raise ValueError("max_tokens must be positive int")
        needed = len(ids) + cap - 1
        if max_model_len is not None and needed > max_model_len:
            raise ValueError(f"{rid}: needs {needed}, model limit {max_model_len}")
        if block_size is not None and num_blocks is not None:
            if needed > block_size * num_blocks:
                raise ValueError(f"{rid}: one request cannot fit KV pool")
        seen.add(rid)
        last = arrival
    if requests[0]["arrival_s"] != 0:
        raise ValueError("first arrival must be zero")
    if trace.get("workload_sha256") != digest(requests):
        raise ValueError("workload digest mismatch")
    return requests


def make_burst(source, *, delay_s=1.0, old_count=4, late_count=2,
               short_length=64, long_length=1024, old_output=128, late_output=16,
               late_interval_s=0.0):
    sizes = (old_count, late_count, short_length, long_length, old_output, late_output)
    if any(type(x) is not int or x < 1 for x in sizes):
        raise ValueError("counts and lengths must be positive integers")
    if type(delay_s) not in (int, float) or not math.isfinite(delay_s) or delay_s <= 0:
        raise ValueError("delay_s must be finite and positive")
    if (type(late_interval_s) not in (int, float) or not math.isfinite(late_interval_s)
            or late_interval_s < 0):
        raise ValueError("late_interval_s must be finite and nonnegative")
    bank = source["requests"]
    if len(bank) < old_count + late_count:
        raise ValueError("not enough source requests")
    requests = []
    for i in range(old_count + late_count):
        old = i < old_count
        base = bank[i]["prompt_token_ids"]
        if not base:
            raise ValueError("empty source prompt")
        length = short_length if old else long_length
        # Exact-length synthetic input: cycle a real tokenized prompt.
        tokens = (base * ((length + len(base) - 1) // len(base)))[:length]
        requests.append({
            "id": f"r{i}", "dataset_index": bank[i]["dataset_index"],
            "group": "old_decode" if old else "late_prefill",
            "arrival_s": 0.0 if old else float(delay_s + (i - old_count) * late_interval_s),
            "prompt_token_ids": tokens,
            "max_tokens": old_output if old else late_output,
        })
    trace = {
        "schema": "online-v1", "synthetic": True,
        "description": "fixed-time long-prompt burst during ongoing generation",
        "model_config_sha256": source["model_config_sha256"],
        "tokenizer_sha256": source["tokenizer_sha256"],
        "requests": requests, "workload_sha256": digest(requests),
    }
    validate_trace(trace)
    return trace


def model_identity(model):
    model = Path(model)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("only Qwen3 is supported")
    return config, {"model_config_sha256": file_hash(model / "config.json"),
                    "tokenizer_sha256": {p.name: file_hash(p) for p in sorted(model.glob("*token*.json"))}}


def check_identity(trace, model):
    config, identity = model_identity(model)
    if any(trace.get(key) != value for key, value in identity.items()):
        raise ValueError("trace model/tokenizer fingerprint does not match --model")
    return config


def make_request_stream(source, options):
    """Preserve real prompt tokens; vary arrival times, never reshape the prompts."""
    selected = source["requests"][options["offset"]:options["offset"] + options["request_count"]]
    if len(selected) != options["request_count"]:
        raise ValueError(f"requested {options['request_count']} requests at offset {options['offset']}, "
                         f"but only {len(selected)} are available")
    if options["shuffle"]:
        random.Random(options["seed"]).shuffle(selected)
    requests, arrival = [], 0.0
    rng = random.Random(options["seed"])
    for i, original in enumerate(selected):
        if i:
            arrival += (1 / options["arrival_rate"] if options["profile"] == "interval"
                        else rng.expovariate(options["arrival_rate"]))
        requests.append({**original, "id": str(original.get("id", original["dataset_index"])),
                         "group": "requests", "arrival_s": arrival, "max_tokens": options["max_tokens"]})
    trace = {"schema": "online-v1", "synthetic": source.get("synthetic", False),
             "description": "original prompts with " + options["profile"] + " open-loop arrivals",
             "model_config_sha256": source["model_config_sha256"],
             "tokenizer_sha256": source["tokenizer_sha256"],
             "requests": requests, "workload_sha256": digest(requests),
             "generation": {k: options[k] for k in ("profile", "seed", "offset", "shuffle", "request_count",
                                                    "arrival_rate", "max_tokens")}}
    return trace


def prepare_dataset(options, identity):
    from transformers import AutoTokenizer
    from experiments.math500.run_math500 import encode_problem

    dataset = Path(options["dataset"])
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    start, count = options["offset"], options["request_count"]
    if len(rows[start:start + count]) != count:
        raise ValueError(f"requested {count} requests at offset {start}, dataset has {len(rows)} rows")
    tokenizer = AutoTokenizer.from_pretrained(options["model"], local_files_only=True)
    requests = []
    for index in range(start, start + count):
        row = rows[index]
        problem = row.get("problem") if isinstance(row, dict) else None
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"dataset row {index} has no nonempty problem")
        requests.append({"id": str(row.get("unique_id", index)), "dataset_index": index,
                         "prompt_token_ids": encode_problem(tokenizer, problem)})
    # The dataset has already been sliced; keep original row indices in requests.
    trace = make_request_stream({**identity, "requests": requests}, {**options, "offset": 0})
    trace["generation"]["offset"] = start
    trace.update(dataset_sha256=file_hash(dataset),
                 template="Qwen user chat, enable_thinking=False, render then encode")
    return trace


def prepare_trace(options):
    """Prepare before timing; dataset mode loads only a local tokenizer, no weights."""
    model = Path(options["model"])
    if options.get("trace"):
        trace = json.loads(Path(options["trace"]).read_text(encoding="utf-8-sig"))
        validate_trace(trace)
        check_identity(trace, model)
        return trace
    config, identity = model_identity(model)
    if options.get("dataset"):
        trace = prepare_dataset(options, identity)
        validate_trace(trace, vocab_size=config["vocab_size"])
        return trace
    count = (options["old_count"] + options["late_count"] if options["profile"] == "burst"
             else options["request_count"])
    if options.get("source"):
        source = json.loads(Path(options["source"]).read_text(encoding="utf-8-sig"))
        if source["workload_sha256"] != digest(source["requests"]):
            raise ValueError("source workload changed")
        check_identity(source, model)
        if options["profile"] != "burst":
            trace = make_request_stream(source, options)
            trace["source_file_sha256"] = file_hash(options["source"])
            validate_trace(trace, vocab_size=config["vocab_size"])
            return trace
        if len(source["requests"]) < count:
            raise ValueError(f"source needs at least {count} requests")
    else:
        rng = random.Random(options["seed"])
        # Synthetic token sequences measure scheduling, not language task quality.
        source = {**identity, "requests": [
            {"dataset_index": i, "prompt_token_ids": [rng.randrange(config["vocab_size"]) for _ in range(128)]}
            for i in range(count)]}
    if options["profile"] == "burst":
        trace = make_burst(source, **{key: options[key] for key in (
            "delay_s", "old_count", "late_count", "short_length", "long_length",
            "old_output", "late_output", "late_interval_s")})
    else:
        rng = random.Random(options["seed"])
        requests, arrival = [], 0.0
        for i in range(count):
            kind = i % len(options["prompt_lengths"])
            length, cap = options["prompt_lengths"][kind], options["output_lengths"][kind]
            if i:
                arrival += (1 / options["arrival_rate"] if options["profile"] == "interval"
                            else rng.expovariate(options["arrival_rate"]))
            base = source["requests"][i]["prompt_token_ids"]
            if not base:
                raise ValueError("empty source prompt")
            requests.append({"id": f"r{i}", "group": f"profile_{kind}",
                             "dataset_index": source["requests"][i]["dataset_index"],
                             "arrival_s": arrival, "max_tokens": cap,
                             "prompt_token_ids": (base * ((length + len(base) - 1) // len(base)))[:length]})
        trace = {"schema": "online-v1", "synthetic": True, **identity,
                 "description": options["profile"] + " open-loop arrivals", "requests": requests,
                 "workload_sha256": digest(requests)}
    trace["generation"] = {key: options[key] for key in (
        "profile", "seed", "delay_s", "old_count", "late_count", "short_length", "long_length",
        "old_output", "late_output", "late_interval_s", "request_count", "arrival_rate",
        "prompt_lengths", "output_lengths")}
    if options.get("source"):
        trace["source_file_sha256"] = file_hash(options["source"])
    validate_trace(trace, vocab_size=config["vocab_size"])
    return trace


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delay-s", type=float, default=1.0)
    parser.add_argument("--long-length", type=int, default=1024)
    parser.add_argument("--short-length", type=int, default=64)
    parser.add_argument("--old-count", type=int, default=4)
    parser.add_argument("--late-count", type=int, default=2)
    parser.add_argument("--old-output", type=int, default=128)
    parser.add_argument("--late-output", type=int, default=16)
    parser.add_argument("--late-interval-s", type=float, default=0)
    args = parser.parse_args(argv)
    source = json.loads(args.source.read_text(encoding="utf-8-sig"))
    if source["workload_sha256"] != digest(source["requests"]):
        raise ValueError("source workload changed")
    trace = make_burst(source, **{key: value for key, value in vars(args).items() if key not in ("source", "out")})
    trace["source_file_sha256"] = file_hash(args.source)
    write_json(args.out, trace)
    print("requests=", len(trace["requests"]), "hash=", trace["workload_sha256"])


if __name__ == "__main__":
    main()
