# TinyLlama ファインチューニング済みモデルの推論スクリプト（CPU版）
# Raspberry Pi 5 など GPU が無い環境用

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========================================
# 設定
# ========================================
MODEL_PATH = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
max_seq_length = 256
dtype = torch.float32

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
    print("train_cpu_simple.py を実行してモデルを作成してください")
    exit(1)

print(f"\nモデルをロード中: {MODEL_PATH}")
try:
    base_model_name = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
    
    # ベースモデルをロード
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    # LoRAアダプタをロード
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print("✓ モデルのロード完了！")
except Exception as e:
    print(f"❌ エラー: {e}")
    exit(1)

# 推論モードに切り替え
model.eval()

# ========================================
# 推論関数
# ========================================
def generate_response(instruction, max_tokens=128, temperature=0.7, top_p=0.8):
    """
    ユーザーの質問に対して回答を生成
    
    Args:
        instruction: ユーザーの質問文
        max_tokens: 最大生成トークン数
        temperature: 生成の多様性（0.0-1.0）
        top_p: nucleus sampling パラメータ
    """
    # プロンプトを組み立て
    prompt = f"User: {instruction}\nAssistant:"
    
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
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # デコード
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # アシスタント部分を抽出
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
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
        response = generate_response(user_input, max_tokens=128)
        print("\r" + " " * 25 + "\r", end="")  # 進捗表示をクリア
        
        print(f"Assistant: {response}\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue
