# 生のUnsloth Qwen3-0.6Bモデルの推論スクリプト（CPU版）
# ファインチューニング未使用、単純版
# Raspberry Pi 5 など GPU が無い環境用

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========================================
# 設定
# ========================================
# CPU版向けの超小型モデルオプション：
# - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B)
# - "microsoft/phi-2" (2.7B) 
# - "gpt2" (124M) - 最軽量
# - "distilgpt2" (82M) - 超軽量
MODEL_NAME = "gpt2"
max_seq_length = 128
dtype = torch.float32

print("="*60)
print("生のUnsloth Qwen3-0.6B推論スクリプト（CPU版）")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU only)'}")
print("="*60)

# ========================================
# モデルのロード
# ========================================
print(f"\nモデルをロード中: {MODEL_NAME}")
try:
    # トークナイザーをロード
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # モデルをロード
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    print("✓ モデルのロード完了！")
except Exception as e:
    print(f"❌ エラー: モデルのロードに失敗しました")
    print(f"詳細: {e}")
    exit(1)

# ========================================
# 推論
# ========================================
def generate_response(prompt: str, max_length: int = 128) -> str:
    """
    プロンプトに基づいてテキストを生成
    """
    print(f"\n📝 プロンプト: {prompt}")
    print("-" * 60)
    
    try:
        # トークン化
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        
        # テキスト生成（CPU環境向けに最適化）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # デコード
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"📤 応答: {response}")
        return response
        
    except Exception as e:
        print(f"❌ エラー: 推論に失敗しました")
        print(f"詳細: {e}")
        return ""

# ========================================
# テスト推論
# ========================================
if __name__ == "__main__":
    # テストプロンプト
    test_prompts = [
        "What is the capital of France?",
        "Hello, how are you?",
        "Write a short poem about the moon.",
    ]
    
    print("\n" + "="*60)
    print("テスト推論を実行中...")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【テスト {i}/{len(test_prompts)}】")
        generate_response(prompt, max_length=128)
        print()
    
    # インタラクティブモード
    print("\n" + "="*60)
    print("インタラクティブモード（終了: 'quit' または 'exit'）")
    print("="*60)
    
    while True:
        user_input = input("\n💬 あなたのプロンプト: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 さようなら！")
            break
        
        if not user_input:
            print("⚠️  プロンプトを入力してください")
            continue
        
        generate_response(user_input, max_length=256)
