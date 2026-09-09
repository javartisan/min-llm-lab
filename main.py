from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
LOCAL_DIR = Path(__file__).resolve().parent / "models" / "SmolLM2-135M"


def is_local_model(path: Path) -> bool:
    return path.exists() and (path / "config.json").exists()


if is_local_model(LOCAL_DIR):
    print(f"从本地加载: {LOCAL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_DIR, local_files_only=True)
else:
    print(f"首次下载并保存到: {LOCAL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID)
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR)
    print(f"已保存到: {LOCAL_DIR}")

text = "Java interface 是什么？"
tokens = tokenizer.tokenize(text)
print(tokens)

# pt 是 PyTorch 的缩写，告诉 tokenizer：把结果做成 PyTorch 的 Tensor，而不是普通 Python 列表。
inputs = tokenizer(text, return_tensors="pt")

print(inputs)


import torch

text = "Java interface 是什么？"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.logits.shape)