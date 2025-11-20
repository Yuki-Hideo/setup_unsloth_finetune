#!/bin/bash
# Qwen2.5 (Qwen3) ファインチューニング環境セットアップ
# 環境: CUDA 12.0 (System) -> CUDA 11.8 (PyTorch) で動作させます
set -e

echo "=========================================="
echo "Unsloth セットアップ (CUDA 12.0環境用)"
echo "Target: CUDA 11.8 Binaries (互換モード)"
echo "=========================================="
echo ""

# 1. 失敗した環境のお掃除 (重要)
echo "[1/4] 既存環境のクリーンアップ..."
pip uninstall -y unsloth unsloth_zoo torch torchvision torchaudio xformers trl peft accelerate bitsandbytes
echo "✓ クリーンアップ完了"
echo ""

# 2. PyTorch 2.4.0 (CUDA 11.8版) のインストール
# CUDA 12.0 の環境では、この 11.8版 を使うのが定石です
echo "[2/4] PyTorch 2.4.0 (CUDA 11.8) インストール中..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch インストール完了"
echo ""

# 3. Unsloth と依存関係のインストール
echo "[3/4] Unsloth (cu118-torch240) インストール中..."

# 依存関係の自動アップグレードによる破損を防ぐため、--no-depsでコアを入れ、その後依存を入れる戦略
pip install "unsloth[cu118-torch240] @ git+https://github.com/unslothai/unsloth.git"

# 必要な依存ライブラリを整合性が取れるようにインストール
pip install --no-deps "xformers<0.0.28" "trl<0.25.0" "peft" "accelerate" "bitsandbytes"
pip install transformers datasets huggingface_hub packaging pyyaml regex requests safetensors tqdm sentencepiece protobuf

echo "✓ Unsloth インストール完了"
echo ""

# 4. 動作確認
echo "[4/4] 動作環境チェック..."
python3 << 'EOF'
import torch
import sys
try:
    print(f"PyTorch Version: {torch.__version__}")

    # CUDAチェック
    if torch.cuda.is_available():
        print(f"CUDA Version (Torch): {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # VRAMチェック
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU VRAM: {vram:.2f} GB")
    else:
        print("!!! 警告: GPUが認識されません !!!")

    # Unslothチェック
    from unsloth import FastLanguageModel
    print("✓ Unsloth Import 成功: 準備完了")

except Exception as e:
    print(f"✗ エラー発生: {e}")
    sys.exit(1)
EOF
