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

# 3. Transformers と依存ライブラリのインストール
echo "[3/4] 依存ライブラリ インストール中..."
# CPU版のトレーニング用ライブラリ
pip install transformers datasets peft trl bitsandbytes accelerate

echo "✓ インストール完了"
echo ""

# 4. 動作確認
echo "[4/4] 動作確認..."
python3 << 'EOF'
import torch
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU only)"}')
print(f'CPU Threads: {torch.get_num_threads()}')

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print('✓ Hugging Face transformers Import OK')
    print('✓ PEFT Import OK')
except Exception as e:
    print(f'⚠ Import Error: {e}')
EOF

echo ""
echo "=========================================="
echo "✓ セットアップ完了！"
echo "=========================================="
echo ""
echo "使用スクリプト:"
echo "  トレーニング: python3 train_cpu.py"
echo "  推論:       python3 inference_cpu.py"
echo ""
echo "詳細は CPU_GUIDE.md を参照してください"
echo "=========================================="
