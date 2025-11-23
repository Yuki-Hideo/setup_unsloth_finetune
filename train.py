# Qwen3-0.6B ファインチューニング with Unsloth
# ローカルマシン用セットアップ完全版

# ========================================
# 事前準備：環境構築
# ========================================
r"""
# 1. Pythonバージョン確認（Python 3.10以上推奨）
python --version

# 2. CUDA対応GPUの確認
nvidia-smi

# 3. 仮想環境の作成（推奨）
python -m venv venv
# Windowsの場合：
venv\Scripts\activate
# Linux/Macの場合：
source venv/bin/activate

# 4. PyTorchのインストール（CUDA 12.1の例）
# https://pytorch.org/ で環境に合わせて選択
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Unslothと依存関係のインストール
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install transformers datasets

# 注：CUDA 11.8の場合は cu118、CUDA 12.4の場合は cu124 を指定
"""

# ========================================
# 1. インポートとGPU確認
# ========================================
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("警告: CUDAが利用できません。CPUモードで実行されます（非常に遅い）")

# ========================================
# 2. モデルとトークナイザーのロード
# ========================================
from unsloth import FastLanguageModel

max_seq_length = 2048  # Qwen3は最大32,768まで対応（メモリに応じて調整）
dtype = None  # 自動検出。float16（Tesla T4、V100）、bfloat16（Ampere+）
load_in_4bit = True  # 4bit量子化でVRAM使用量を削減

print("\nモデルをロード中...")
# モデルのロード（初回は数GB自動ダウンロード）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",  # 4bit量子化済みモデル
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="hf_...",  # Gated modelの場合は必要
)
print("モデルのロード完了！")

# ========================================
# 3. LoRA設定（高速ファインチューニング）
# ========================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRAのランク（8, 16, 32, 64など）
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0,  # 最適化のため0を推奨
    bias="none",
    use_gradient_checkpointing="unsloth",  # メモリ効率向上
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ========================================
# 4. データセットの準備
# ========================================
from datasets import load_dataset

# 例：Alpacaデータセット（他のデータセットに変更可能）
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Qwen3のチャット形式テンプレート
# enable_thinking=False の場合（通常の対話）
alpaca_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}{}<|im_end|>
<|im_start|>assistant
{}{}"""

EOS_TOKEN = tokenizer.eos_token  # </s>

def formatting_prompts_func(examples):
    """データセットをQwen3形式にフォーマット"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    
    for instruction, input, output in zip(instructions, inputs, outputs):
        # inputがある場合は追加
        text_input = f"\n{input}" if input else ""
        text = alpaca_prompt.format(instruction, text_input, output, EOS_TOKEN)
        texts.append(text)
    
    return {"text": texts}

# データセットをフォーマット
dataset = dataset.map(formatting_prompts_func, batched=True)

# ========================================
# 5. トレーニング設定
# ========================================
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # 短いシーケンスには有効
    args=TrainingArguments(
        per_device_train_batch_size=4,  # RTX 3090 Tiなら4に増やせる
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=1000,  # 実際は数百〜数千に設定
        # num_train_epochs=1,  # max_stepsの代わりにepoch数を指定可能
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # "wandb"に変更でWandB連携
    ),
)

# ========================================
# 6. トレーニング実行
# ========================================
print("\n" + "="*50)
print("トレーニング開始...")
print("="*50)

# トレーニング実行
trainer_stats = trainer.train()

print("\n" + "="*50)
print("トレーニング完了！")
print(f"最終Loss: {trainer_stats.training_loss:.4f}")
print("="*50)

# ========================================
# 7. モデルの保存
# ========================================
import os

print("\nモデルを保存中...")

# 保存ディレクトリを作成
os.makedirs("./finetuned_models", exist_ok=True)

# LoRAモデルを保存（軽量、再学習に便利）
print("LoRAアダプタを保存中...")
model.save_pretrained("./finetuned_models/lora_model")
tokenizer.save_pretrained("./finetuned_models/lora_model")

# 16bit完全マージモデルを保存（推論用、高精度）
print("16bit完全マージモデルを保存中...")
model.save_pretrained_merged(
    "./finetuned_models/merged_16bit", 
    tokenizer, 
    save_method="merged_16bit"
)

# 4bit量子化モデルを保存（軽量推論用）
print("4bit量子化モデルを保存中...")
model.save_pretrained_merged(
    "./finetuned_models/merged_4bit", 
    tokenizer, 
    save_method="merged_4bit"
)

# GGUF形式で保存（llama.cppやOllama用）
print("GGUF形式で保存中...")
model.save_pretrained_gguf(
    "./finetuned_models/gguf", 
    tokenizer, 
    quantization_method="q4_k_m"
)

print("\n保存完了！以下のディレクトリに保存されました：")
print("  - ./finetuned_models/lora_model (LoRAアダプタ)")
print("  - ./finetuned_models/merged_16bit (16bit完全モデル)")
print("  - ./finetuned_models/merged_4bit (4bit量子化モデル)")
print("  - ./finetuned_models/gguf (GGUF形式)")

# ========================================
# 8. 推論テスト
# ========================================
# 推論モードに切り替え
FastLanguageModel.for_inference(model)

# テストプロンプト
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "日本の首都はどこですか？",  # instruction
            "",  # input
            "",  # output（空にして生成）
            ""   # EOS（生成時は不要）
        )
    ],
    return_tensors="pt"
).to("cuda")

# テキスト生成
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

outputs = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=128,
    use_cache=True,
    # Qwen3推奨設定（非Thinkingモード）
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
)

print("\n推論テスト完了！")

# ========================================
# 9. Hugging Face Hubへアップロード（オプション）
# ========================================
"""
model.push_to_hub("your_username/qwen3-0.6b-finetuned", token="hf_...")
tokenizer.push_to_hub("your_username/qwen3-0.6b-finetuned", token="hf_...")
"""

# ========================================
# 10. その他のエクスポート形式
# ========================================
"""
# vLLM用
model.save_pretrained_merged("model_vllm", tokenizer, save_method="merged_16bit")

# Ollamaへエクスポート
model.push_to_hub_gguf(
    "your_username/qwen3-0.6b-finetuned-gguf",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0"],
    token="hf_..."
)
"""
