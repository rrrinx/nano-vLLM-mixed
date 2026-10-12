# Sources and Third-Party Notices

[中文](docs/zh-CN/THIRD_PARTY_NOTICES.md)

- The inference-engine architecture references GeeeekExplorer/nano-vllm. Its original MIT copyright and license text is retained in `licenses/nano-vllm-MIT.txt`. This notice does not automatically license the author's additions under the same terms.
- Hugging Face Transformers' Qwen3 implementation provides the numerical reference; models are loaded from local checkpoints. Model weights and tokenizers are not distributed in this repository.
- Experiment inputs come from NVIDIA SPEED-Bench `throughput_2k`. Parquet files, source text, and decodable input-token sequences are not distributed. Published copies contain only sample indices, lengths, fingerprints, and performance measurements. Inference reproduction requires obtaining the dataset separately.
- This publication preparation does not select a new open-source license for the project's own code. The maintainer may select one when publishing; upstream notices must be retained.
