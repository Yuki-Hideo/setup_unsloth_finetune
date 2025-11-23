# Qwen3-0.6B ファインチューニング with Unsloth (CPU版)
# Raspberry Pi 5 など GPU が無い環境用

# ========================================
# 事前準備
# ========================================
r"""
# Raspberry Pi 5 での実行方法:
# 1. セットアップスクリプトを実行
chmod +x scripts/setup_raspberry_pi.sh
./scripts/setup_raspberry_pi.sh

# 2. JSONデータセットを準備
# data/famicom_dataset.json を配置

# 3. トレーニング実行（低メモリ設定）
python3 train_cpu.py

# 注: CPU版のため非常に遅い。大規模モデルは推奨されません。
"""

# ========================================
# 1. インポートとシステム確認
# ========================================
import torch
import os
import json
from pathlib import Path

print("="*60)
print("Unsloth CPU版トレーニング (Raspberry Pi対応)")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU Threads: {torch.get_num_threads()}")
print(f"Memory: {os.popen('free -h').read().split(chr(10))[1]}")
print("="*60)

# ========================================
# 2. モデルとトークナイザーのロード
# ========================================
from unsloth import FastLanguageModel

max_seq_length = 512  # CPU版は短めに（メモリ節約）
dtype = torch.float32  # CPU では float32 が必須
load_in_4bit = False  # CPU版は量子化不可

print("\nモデルをロード中... (初回は数分かかります)")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-0.5B",  # より小さいモデル
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    print("✓ モデルのロード完了！")
except Exception as e:
    print(f"エラー: {e}")
    print("→ インターネット接続を確認してください")
    exit(1)

# ========================================
# 3. LoRA設定（CPU向け最小限設定）
# ========================================
model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # CPU版は小さめに
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,  # CPU版では不要
    random_state=3407,
)

print("✓ LoRA設定完了")

# ========================================
# 4. データセットの準備
# ========================================
from datasets import Dataset

data_path = Path("data/famicom_dataset.json")

if not data_path.exists():
    print(f"\n⚠ データセットが見つかりません: {data_path}")
    print("サンプルデータセットを作成します...")
    
    # サンプルデータを作成
    sample_data = [
        {
            "instruction": "What is the Famicom Disk System?",
            "input": "",
            "output": "The Famicom Disk System is a peripheral for Nintendo's Family Computer home video game console."
        },
        {
            "instruction": "When was the Famicom Disk System released?",
            "input": "",
            "output": "The Famicom Disk System was released in Japan on February 21, 1986."
        }
    ]
    
    os.makedirs("data", exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"✓ サンプルデータを作成: {data_path}")
else:
    print(f"✓ データセットを読み込み: {data_path}")

# JSONファイルを読み込む
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_dict({
    "instruction": [item["instruction"] for item in data],
    "input": [item.get("input", "") for item in data],
    "output": [item["output"] for item in data]
})

print(f"  - データ数: {len(dataset)}")

# Qwen3 チャット形式テンプレート
alpaca_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}{}<|im_end|>
<|im_start|>assistant
{}{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    """Qwen3形式にフォーマット"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text_input = f"\n{input_text}" if input_text else ""
        text = alpaca_prompt.format(instruction, text_input, output, EOS_TOKEN)
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ========================================
# 5. トレーニング設定（CPU向け最小リソース版）
# ========================================
from trl import SFTTrainer
from transformers import TrainingArguments

# CPU版向けの最小限な設定
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # CPU版は必ず1
    gradient_accumulation_steps=1,
    warmup_steps=2,
    max_steps=10,  # 少なめから開始
    learning_rate=2e-4,
    fp16=False,  # CPU版は float32 のみ
    bf16=False,
    logging_steps=1,
    optim="adamw_torch",  # CPU版では torch 版
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
    save_strategy="steps",
    save_steps=5,
    dataloader_num_workers=0,  # CPU版は並列不可
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

# ========================================
# 6. トレーニング実行
# ========================================
print("\n" + "="*60)
print("⏱  トレーニング開始（CPU版のため時間がかかります）")
print("="*60)

try:
    trainer_stats = trainer.train()
    print("\n" + "="*60)
    print("✓ トレーニング完了！")
    print(f"最終Loss: {trainer_stats.training_loss:.4f}")
    print("="*60)
except KeyboardInterrupt:
    print("\n⚠ トレーニングが中断されました")
except Exception as e:
    print(f"\n❌ エラー: {e}")
    exit(1)

# ========================================
# 7. モデルの保存
# ========================================
print("\nモデルを保存中...")

os.makedirs("./finetuned_models", exist_ok=True)

# LoRAモデルを保存（CPU版推奨）
print("1. LoRA アダプタを保存中...")
model.save_pretrained("./finetuned_models/lora_model")
tokenizer.save_pretrained("./finetuned_models/lora_model")
print("   ✓ 保存完了: ./finetuned_models/lora_model")

# 32bit完全マージモデルを保存（CPU推論用）
print("2. 完全マージモデルを保存中...")
try:
    model.save_pretrained_merged(
        "./finetuned_models/merged_32bit", 
        tokenizer, 
        save_method="merged_16bit"
    )
    print("   ✓ 保存完了: ./finetuned_models/merged_32bit")
except Exception as e:
    print(f"   ⚠ マージ保存失敗: {e}")
    print("   → LoRA版で推論してください")

print("\n" + "="*60)
print("✓ ファインチューニング完了！")
print("="*60)
print("\n推論スクリプト:")
print("  python3 inference_cpu.py")
print("="*60)
