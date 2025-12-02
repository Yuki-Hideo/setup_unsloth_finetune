## Train ファイル（3種類）

| ファイル | 用途 | 特徴 |
|---------|------|------|
| train.py | GPU環境用 | Unsloth使用、高速、CUDA必須、大規模データ対応 |
| train_cpu.py | Raspberry Pi用（Unsloth） | CPU版、Unsloth + LoRA、メモリ節約 |
| train_cpu_simple.py | Raspberry Pi用（Transformers） | CPU版、Transformers + PEFT、最もシンプル |

## Inference ファイル（5種類）

| ファイル | 用途 | 特徴 |
|---------|------|------|
| inference.py | GPU推論 | Unsloth使用、高速、ファインチューニング済みモデル対応 |
| inference_cpu.py | Raspberry Pi推論 | Qwen3-0.6B + LoRA |
| inference_cpu_simple.py | Raspberry Pi推論 | TinyLlama + LoRA（現在開いてるファイル） |
| raw_model_inference_cpu_simple.py | ベースモデルのみ | ファインチューニング未使用、gpt2など軽量モデル |
| withoutFTinference.py | GPU推論（未微調整） | ベースの Qwen3-0.6B を直接使用 |
| withoutFTinference_cpu_simple.py | CPU推論（未微調整） | ベースの Qwen3-0.6B を CPU で直接使用 |