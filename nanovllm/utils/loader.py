import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from ..models.qwen3 import Qwen3ForCausalLM

DTYPES = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def check_config(config):
    # 这几项在自写模型里是写死的，恰好与 Qwen3-0.6B 的取值一致，所以不检查也看不出问题；
    # 直到指向一个开了它们的模型，那时每个 logit 都是错的，而键集检查发现不了。
    if config.get("attention_bias", False):
        raise ValueError(f"attention_bias={config['attention_bias']}：本实现的 Linear 全部 bias=False")
    if config.get("rope_scaling") is not None:
        raise ValueError(f"rope_scaling={config['rope_scaling']}：RotaryEmbedding 未实现任何缩放")
    if config.get("sliding_window") is not None or config.get("use_sliding_window", False):
        raise ValueError(
            f"sliding_window={config.get('sliding_window')}：注意力 mask 是纯因果，没有窗口"
        )
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError(f"hidden_act={config['hidden_act']}：SwiGLU 写死了 silu")


def resolve_head_dim(config):
    # 必须用声明值：Qwen3-0.6B 是 128，而 hidden_size // num_attention_heads = 64。
    declared = config.get("head_dim")
    if declared is not None:
        return declared
    return config["hidden_size"] // config["num_attention_heads"]


def to_torch_dtype(name):
    if isinstance(name, torch.dtype):
        return name
    try:
        return DTYPES[name]
    except KeyError:
        raise ValueError(f"未知的 dtype: {name!r}，只支持 {sorted(DTYPES)}") from None


def resolve_dtype(config, dtype):
    if dtype is not None:
        return to_torch_dtype(dtype)
    # transformers >= 4.56 把 torch_dtype 改名成 dtype，老 checkpoint 只有前者。
    return to_torch_dtype(config.get("dtype") or config.get("torch_dtype") or "float32")


def resolve_checkpoint_files(model_dir):
    """Return weight paths and the shard index's weight_map (None for one file).

    Check every indexed shard before loading tensors. HF cache snapshots may
    use symlinks, so validate the relative filename without resolving symlinks.
    """
    directory = Path(model_dir)
    single = directory / "model.safetensors"
    if single.is_file():
        return [single], None

    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"模型目录 {directory} 中没有 model.safetensors 或 model.safetensors.index.json；"
            "分片权重需要索引文件及其引用的全部 .safetensors 文件"
        )
    with index_path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (not isinstance(weight_map, dict) or not weight_map
            or any(not isinstance(name, str) or not name
                   or not isinstance(filename, str) or not filename
                   for name, filename in weight_map.items())):
        raise ValueError(f"权重索引 {index_path} 的 weight_map 必须是非空的 tensor 名称到分片文件名的映射")

    paths = []
    for filename in sorted(set(weight_map.values())):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
            raise ValueError(f"权重索引 {index_path} 中的分片文件名无效：{filename}")
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"权重索引 {index_path} 引用的分片不存在：{path}；请补齐模型文件")
        if path not in paths:
            paths.append(path)
    return paths, weight_map


def load_state_dict_from_dir(model_dir):
    paths, weight_map = resolve_checkpoint_files(model_dir)
    state_dict = {}
    for path in paths:
        shard = load_file(str(path), device="cpu")
        if weight_map is not None:
            expected = {name for name, filename in weight_map.items()
                        if Path(model_dir) / filename == path}
            missing, unexpected = expected - shard.keys(), shard.keys() - expected
            if missing or unexpected:
                raise RuntimeError(
                    f"分片 {path} 与权重索引不一致：缺少 {sorted(missing)}，多余 {sorted(unexpected)}"
                )
        # Index validation ensures disjoint keys; update retains the original
        # tensor objects and checkpoint dtypes, without copying shard tensors.
        state_dict.update(shard)
    return state_dict


def fill_tied_lm_head(config, state_dict):
    """HF 自己 save_pretrained 出来的 tied checkpoint 不写 lm_head.weight。

    只补齐这一个白名单键，别的缺失照旧报错。塞进去的必须是同一个张量对象而不是副本，
    load_state_dict 的 copy_ 才会把两个名字写进同一块 storage，别名关系才保得住。

    文件同时提供两个名字时（Qwen3-0.6B 就是），它们最终指向同一个 Parameter，
    加载顺序决定谁最后写入，不一致会被静默覆盖，所以先断言相等。
    """
    if not config.get("tie_word_embeddings", False):
        return state_dict

    if "lm_head.weight" not in state_dict:
        if "model.embed_tokens.weight" not in state_dict:
            raise KeyError("tied checkpoint 两个名字都没有，无法补齐 lm_head.weight")
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    elif not torch.equal(state_dict["lm_head.weight"], state_dict["model.embed_tokens.weight"]):
        raise ValueError("lm_head.weight 与 model.embed_tokens.weight 不一致，tied checkpoint 不应如此")

    return state_dict


def load_model(model_dir, kv_cache_manager, device="cuda",  dtype=None, model_cls = Qwen3ForCausalLM):
    config = load_config(model_dir)
    check_config(config)

    model = model_cls(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_hidden_layers'],
        heads_num=config['num_attention_heads'],
        heads_kv_num=config['num_key_value_heads'],
        heads_dim=resolve_head_dim(config),
        max_position_embeddings=config['max_position_embeddings'],
        base=config['rope_theta'],
        intermediate_size=config['intermediate_size'],
        rms_norm_eps=config['rms_norm_eps'],
        tie_word_embeddings=config['tie_word_embeddings'],
        kv_cache_manager=kv_cache_manager,
    )

    hf_state_dict = fill_tied_lm_head(config, load_state_dict_from_dir(model_dir))

    # bf16 的 checkpoint 装进 fp32 参数是无损加宽（copy_ 自带转换），最后再一次性降到
    # 目标 dtype；反过来先把权重降精度再加载才会丢东西。
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"权重不匹配：缺少 {missing}，多余 {unexpected}")

    model.to(device=device, dtype=resolve_dtype(config, dtype))
    model.eval()
    model.requires_grad_(False)

    return model
