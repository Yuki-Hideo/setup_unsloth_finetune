#!/bin/bash
set -e

echo "=========================================="
echo "Unsloth 安定版セットアップ (RTX 3090 Ti用)"
echo "Target: PyTorch 2.4.0 + CUDA 12.1"
echo "=========================================="

# 1. 環境のリセット
echo "[1/4] クリーンアップ中..."
pip uninstall -y unsloth unsloth_zoo torch torchvision torchaudio xformers trl peft accelerate bitsandbytes
echo "✓ 完了"
echo ""

# 2. PyTorch 2.4.0 (CUDA 12.1) のインストール
# ※ 3090Tiにとって最も安定しており、Unslothのテストが最も充実しているバージョンです
echo "[2/4] PyTorch 2.4.0 (cu121) インストール中..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch インストール完了"
echo ""

echo "[3/4] Unsloth (cu121-torch240) インストール中..."
pip install unsloth

# 依存ライブラリの整合性確保
pip install --no-deps "xformers<0.0.28" "trl<0.25.0" "peft" "accelerate" "bitsandbytes"

echo "✓ Unsloth インストール完了"
echo ""

# 4. 動作確認
echo "[4/4] 動作確認..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
from unsloth import FastLanguageModel
print('✓ Unsloth Import OK')
"
