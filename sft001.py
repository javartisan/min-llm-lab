from pathlib import Path

# torch 2.2.2 (Intel Mac) 没有 is_macos_or_newer；transformers 检查 bf16 时会崩
import torch.backends.mps as _mps
if not hasattr(_mps, "is_macos_or_newer"):
    def _is_macos_or_newer(major: int, minor: int = 0) -> bool:
        if major > 13:
            return False
        if major == 13:
            return _mps.is_macos13_or_newer(minor)
        return True

    _mps.is_macos_or_newer = _is_macos_or_newer

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)

training_args = SFTConfig(
    output_dir="./checkpoints/smollm2-135m-java",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100,
    report_to="none",
    bf16=False,
    fp16=False,
    loss_type="nll",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model(
    "./models/smollm2-135m-java"
)

tokenizer.save_pretrained(
    "./models/smollm2-135m-java"
)
