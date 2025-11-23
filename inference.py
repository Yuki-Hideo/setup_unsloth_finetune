# Qwen3-0.6B ファインチューニング済みモデルの推論スクリプト
# トレーニング後に別プロセスで推論を行う場合に使用

import torch
from unsloth import FastLanguageModel

# ========================================
# 設定
# ========================================
# 使用するモデルのパスを指定
MODEL_PATH = "./finetuned_models/merged_16bit"  # または merged_16bit, lora_model
max_seq_length = 2048
dtype = None
load_in_4bit = Falser

# ========================================
# モデルのロード
# ========================================
print("モデルをロード中...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 推論モードに切り替え
FastLanguageModel.for_inference(model)
print("モデルのロード完了！\n")

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
        temperature: 生成のランダム性（0.0-1.0）
        top_p: Top-p sampling
    """
    # プロンプトの準備
    prompt = alpaca_prompt.format(instruction, "")
    
    # トークン化
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # テキスト生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            top_k=20,
            min_p=0.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # デコード（入力部分を除去）
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()

# ========================================
# インタラクティブモード
# ========================================
def interactive_chat():
    """対話形式でモデルと会話"""
    print("="*60)
    print("Qwen3-0.6B ファインチューニング済みモデル - 対話モード")
    print("="*60)
    print("終了するには 'quit', 'exit', 'q' を入力してください\n")
    
    while True:
        try:
            # ユーザー入力
            user_input = input("あなた: ").strip()
            
            # 終了コマンド
            if user_input.lower() in ['quit', 'exit', 'q', '終了']:
                print("終了します。")
                break
            
            # 空入力をスキップ
            if not user_input:
                continue
            
            # 回答生成
            print("AI: ", end="", flush=True)
            response = generate_response(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}\n")

# ========================================
# テスト実行
# ========================================
def test_examples():
    """いくつかの例で動作テスト"""
    print("="*60)
    print("Qwen3-0.6B - テスト実行")
    print("="*60)
    
    test_prompts = [
        "日本の首都はどこですか？",
        "Pythonでリストの要素を反転する方法を教えてください。",
        "人工知能とは何ですか？簡潔に説明してください。",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【テスト {i}】")
        print(f"質問: {prompt}")
        print(f"回答: {generate_response(prompt, max_tokens=200)}")
        print("-" * 60)

# ========================================
# メイン実行
# ========================================
if __name__ == "__main__":
    import sys
    
    # コマンドライン引数でモードを選択
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # テストモード
            test_examples()
        elif sys.argv[1] == "chat":
            # 対話モード
            interactive_chat()
        else:
            print("使用方法:")
            print("  python inference.py test  # テスト実行")
            print("  python inference.py chat  # 対話モード")
    else:
        # デフォルトは対話モード
        interactive_chat()

"""
使用例：
1. テスト実行: python inference.py test
2. 対話モード: python inference.py chat (またはデフォルト)
3. スクリプト内で使用:
   
   from inference import generate_response
   answer = generate_response("あなたの質問")
   print(answer)
"""
