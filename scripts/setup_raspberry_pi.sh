#!/bin/bash
set -e

echo "=========================================="
echo "Unsloth セットアップ (Raspberry Pi 5用)"
echo "Target: PyTorch CPU + Python 3.11.2"
echo "Memory: 8GB RAM"
echo "=========================================="

# 1. 環境のリセット
echo "[1/4] クリーンアップ中..."
pip uninstall -y unsloth unsloth_zoo torch torchvision torchaudio xformers trl peft accelerate bitsandbytes 2>/dev/null || true
echo "✓ 完了"
echo ""

# 2. PyTorch CPU版のインストール
# ※ ARM対応のCPU専用バージョン
echo "[2/4] PyTorch (CPU版) インストール中..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "✓ PyTorch インストール完了"
echo ""

# 3. Unsloth と依存ライブラリのインストール
echo "[3/4] Unsloth インストール中..."
pip install unsloth

# メモリ効率重視の依存ライブラリ設定
pip install --no-deps "xformers<0.0.28" "trl<0.25.0" "peft" "accelerate" "bitsandbytes"

echo "✓ Unsloth インストール完了"
echo ""

# 4. 動作確認
echo "[4/4] 動作確認..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None (CPU only)\"}')
print(f'CPU Threads: {torch.get_num_threads()}')
from unsloth import FastLanguageModel
print('✓ Unsloth Import OK')
"

echo ""
echo "=========================================="
echo "✓ セットアップ完了！"
echo "=========================================="
echo ""
echo "推奨設定:"
echo "  - 学習時は --load_in_8bit フラグを使用"
echo "  - バッチサイズは 1 or 2 推奨"
echo "  - gradient_accumulation_steps を使用してメモリ節約"
echo "=========================================="
