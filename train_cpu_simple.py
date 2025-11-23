# Qwen3-0.6B ファインチューニング with Transformers (CPU版)
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
python3 train_cpu_simple.py

# 注: CPU版のため非常に遅い。大規模モデルは推奨されません。
"""

# ========================================
# 1. インポートとシステム確認
# ========================================
import torch
import os
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

print("="*60)
print("Transformers CPU版トレーニング (Raspberry Pi対応)")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU Threads: {torch.get_num_threads()}")
print(f"Memory: {os.popen('free -h').read().split(chr(10))[1]}")
print("="*60)

# ========================================
# 2. モデルとトークナイザーのロード
# ========================================
max_seq_length = 256  # CPU版は短めに（メモリ節約）
dtype = torch.float32  # CPU では float32 が必須

print("\nモデルをロード中... (初回は数分かかります)")
try:
    # TinyLlama: 1.1B の軽量モデル
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("✓ モデルのロード完了！")
except Exception as e:
    print(f"❌ エラー: {e}")
    print("→ インターネット接続を確認してください")
    exit(1)

# ========================================
# 3. LoRA設定（CPU向け最小限設定）
# ========================================
print("\nLoRA設定中...")
lora_config = LoraConfig(
    r=8,  # CPU版は小さめに
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # TinyLlamaの注意層
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✓ LoRA設定完了")

# ========================================
# 4. データセットの準備
# ========================================
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
        },
        {
            "instruction": "How many games were released for the Famicom Disk System?",
            "input": "",
            "output": "There are about 194 games for the Famicom Disk System."
        },
        {
            "instruction": "What are popular games in the Famicom Disk System?",
            "input": "",
            "output": "The popular games include The Super Mario Bros. 2, The Legend of Zelda, and Metroid."
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

# 会話形式のテンプレート
chat_template = """User: {instruction}{input}
Assistant: {output}"""

texts = []
for item in data:
    instruction = item["instruction"]
    input_text = f"\n{item['input']}" if item.get("input", "") else ""
    output = item["output"]
    text = chat_template.format(
        instruction=instruction,
        input=input_text,
        output=output
    )
    texts.append(text)

# データセット作成
dataset = Dataset.from_dict({"text": texts})
print(f"✓ データセット作成完了 - {len(dataset)}件")

# ========================================
# 5. トークナイザー設定
# ========================================
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    """テキストをトークン化"""
    return tokenizer(
        examples["text"],
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
    )

# トークン化
print("\nトークン化中...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"],
)
print(f"✓ トークン化完了")

# ========================================
# 6. トレーニング設定（CPU向け最小リソース版）
# ========================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果言語モデリング
)

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,  # CPU版は必ず1
    gradient_accumulation_steps=1,
    warmup_steps=1,
    max_steps=5,  # サンプルは少なめ
    learning_rate=2e-4,
    fp16=False,  # CPU版は float32 のみ
    bf16=False,
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_strategy="steps",
    save_steps=5,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ========================================
# 7. トレーニング実行
# ========================================
print("\n" + "="*60)
print("⏱  トレーニング開始（CPU版のため時間がかかります）")
print("="*60)

try:
    trainer_stats = trainer.train()
    print("\n" + "="*60)
    print("✓ トレーニング完了！")
    print("="*60)
except KeyboardInterrupt:
    print("\n⚠ トレーニングが中断されました")
except Exception as e:
    print(f"\n❌ エラー: {e}")
    exit(1)

# ========================================
# 8. モデルの保存
# ========================================
print("\nモデルを保存中...")

os.makedirs("./finetuned_models", exist_ok=True)

# LoRAモデルを保存（CPU版推奨）
print("1. LoRA アダプタを保存中...")
model.save_pretrained("./finetuned_models/lora_model")
tokenizer.save_pretrained("./finetuned_models/lora_model")
print("   ✓ 保存完了: ./finetuned_models/lora_model")

# 完全モデルをマージして保存
print("2. 完全マージモデルを保存中...")
try:
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("./finetuned_models/merged_model")
    tokenizer.save_pretrained("./finetuned_models/merged_model")
    print("   ✓ 保存完了: ./finetuned_models/merged_model")
except Exception as e:
    print(f"   ⚠ マージ保存失敗: {e}")
    print("   → LoRA版で推論してください")

print("\n" + "="*60)
print("✓ ファインチューニング完了！")
print("="*60)
print("\n推論スクリプト:")
print("  python3 inference_cpu_simple.py")
print("="*60)
