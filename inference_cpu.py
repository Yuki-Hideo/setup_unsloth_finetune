# Qwen3-0.6B ファインチューニング済みモデルの推論スクリプト（CPU版）
# Raspberry Pi 5 など GPU が無い環境用

import torch
from pathlib import Path
from unsloth import FastLanguageModel

# ========================================
# 設定
# ========================================
MODEL_PATH = "./finetuned_models/lora_model"  # LoRA版推奨（CPU版）
max_seq_length = 512  # CPU版は短めに
dtype = torch.float32  # CPU では float32 が必須
load_in_4bit = False

print("="*60)
print("CPU版推論スクリプト")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU only)'}")
print("="*60)

# ========================================
# モデルのロード
# ========================================
model_dir = Path(MODEL_PATH)

if not model_dir.exists():
    print(f"❌ エラー: モデルが見つかりません: {MODEL_PATH}")
    print("train_cpu.py を実行してモデルを作成してください")
    exit(1)

print(f"\nモデルをロード中: {MODEL_PATH}")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    print("✓ モデルのロード完了！")
except Exception as e:
    print(f"❌ エラー: {e}")
    exit(1)

# 推論モードに切り替え
FastLanguageModel.for_inference(model)

# ========================================
# チャット形式のテンプレート
# ========================================
alpaca_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}"""

# ========================================
# 推論関数
# ========================================
def generate_response(instruction, max_tokens=256, temperature=0.7, top_p=0.8):
    """
    ユーザーの質問に対して回答を生成
    
    Args:
        instruction: ユーザーの質問文
        max_tokens: 最大生成トークン数
        temperature: 生成の多様性（0.0-1.0）
        top_p: nucleus sampling パラメータ
    """
    # プロンプトを組み立て
    prompt = alpaca_prompt.format(instruction, "")
    
    # トークナイズ
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 生成
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    # デコード
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # アシスタント部分を抽出
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response

# ========================================
# インタラクティブチャット
# ========================================
print("\n" + "="*60)
print("インタラクティブ推論モード")
print("'quit' または 'exit' で終了")
print("="*60)
print()

while True:
    try:
        # ユーザー入力
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        # 生成
        print("\nGenerating response...", end="", flush=True)
        response = generate_response(user_input, max_tokens=256)
        print("\r" + " " * 25 + "\r", end="")  # 進捗表示をクリア
        
        print(f"Assistant: {response}\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue
