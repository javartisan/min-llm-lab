"""Day 3 对照：135M 全量微调正式脚本（不用 LoRA）。

学什么
    全量微调 = 底座每一个 requires_grad=True 的参数都会被 optimizer 更新。
    和 scripts/sft001.py 的差别只有一处关键：SFTTrainer 不传 peft_config。
    数据格式、SFTConfig、chat_template 都与 LoRA 版相同，方便对照。

对应阅读
    scripts/sft001.py（LoRA 正式训练）
    docs/学习路线/材料/微调学习笔记.md 第 6 节

运行（仓库根目录）
    python learn/week03/09_full_sft_135m.py
    python learn/week03/09_full_sft_135m.py --epochs 2
    python learn/week03/09_full_sft_135m.py --max-steps 20   # 只跑很少步，先确认能训

产物（不会覆盖 LoRA 目录）
    checkpoints/smollm2-135m-java-full/
    models/smollm2-135m-java-full/

----------------------------------------------------------------------
科普：为什么另开一份，而不是改 sft001

    sft001 是本仓库推荐的 Java 问答训练入口（LoRA）。
    全量微调在小数据上更容易把那几条样本背下来，也更吃优化器内存。
    所以默认仍用 LoRA；本脚本只为让你亲手跑通「不冻底座」的正式循环。

    学习率通常更小：sft001 的 LoRA 用 1e-4；全量常用 2e-5 左右。
    epoch 也更少：LoRA 可多学几轮，全量 2～3 轮往往就够（再多容易过拟合）。

    保存时不需要 merge：改的就是完整 W，save_pretrained 即完整模型。
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from _common import (
    ROOT,
    TRAIN_FILE,
    append_run,
    count_params,
    find_model_dir,
    pct,
    pick_device,
)

CHECKPOINT_DIR = ROOT / "checkpoints" / "smollm2-135m-java-full"
OUTPUT_DIR = ROOT / "models" / "smollm2-135m-java-full"

SYSTEM_PROMPT = (
    "你是一个简洁的 Java 助教。请用一两句中文准确回答，不要跑题，不要输出无关代码。"
)


def to_messages(example: dict) -> dict:
    """把一行 {prompt, completion} 转成 Instruct 习惯的 messages。

    与 sft001.py 相同：system 角色 + user 问题 + assistant 标准答案。
    SFTTrainer 见到 messages 会用 chat_template 拼训练文本。
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


def build_dataset():
    """加载 train.jsonl，map 成 messages，并打印条数和第一条示例。"""
    print(f"[2/5] 加载训练数据: {TRAIN_FILE}")
    raw = load_dataset("json", data_files=str(TRAIN_FILE), split="train")
    dataset = raw.map(to_messages, remove_columns=raw.column_names)
    print(f"      样本数: {len(dataset)}")
    print("      示例 messages:")
    for msg in dataset[0]["messages"]:
        print(f"        - {msg['role']}: {msg['content']}")
    return dataset


def build_training_args(*, epochs: int, max_steps: int, lr: float) -> SFTConfig:
    """组装全量微调的 SFTConfig：学习率更小、默认 epoch 更少。

    max_steps>0 时按步数早停，方便 --max-steps 冒烟。
    有效 batch ≈ per_device_train_batch_size × gradient_accumulation_steps。
    """
    print("[3/5] 配置训练超参数（全量：学习率更小、轮数更少）")
    kwargs: dict = {}
    if max_steps > 0:
        kwargs["max_steps"] = max_steps
        print(f"      注意：--max-steps={max_steps} 会覆盖 epoch 提前停")
    return SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        bf16=False,
        fp16=False,
        loss_type="nll",
        packing=False,
        eos_token="<|im_end|>",
        **kwargs,
    )


def train(args: argparse.Namespace) -> None:
    """正式训练循环：加载 135M Instruct → 数参数（应约 100%）→ SFTTrainer.train → 保存。

    不传 peft_config，所以更新全部权重。保存到 models/smollm2-135m-java-full/，
    不覆盖 LoRA 产物。结束时往 week03_runs.jsonl 写一条记录。
    """
    device = pick_device()
    instruct = ROOT / "models" / "SmolLM2-135M-Instruct"
    model_dir = instruct if (instruct / "config.json").exists() else find_model_dir("135m")
    print(f"[1/5] 加载 135M Instruct 底座: {model_dir}  → 将由 Trainer 放到 {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    total, trainable = count_params(model)
    print(f"      总参数 {total:,}  可训练 {trainable:,}  ({pct(trainable, total):.2f}%)")
    print("      没有 peft_config → 这就是全量微调：几乎 100% 参数都会被更新。")
    print("      对照：scripts/sft001.py 传入 LoraConfig 后，可训练大约只剩几个百分点。")

    dataset = build_dataset()
    training_args = build_training_args(
        epochs=args.epochs, max_steps=args.max_steps, lr=args.lr
    )

    print("[4/5] 开始 SFT 训练（全量，无 LoRA）...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        # 故意不传 peft_config：与 sft001.py 的唯一关键差别
    )
    train_result = trainer.train()
    loss = float(train_result.training_loss)
    print(f"      train_loss = {loss:.4f}")

    print(f"[5/5] 保存完整模型（无需 merge）: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"      权重目录: {OUTPUT_DIR}")
    print("      评测可把路径指到这里；默认 eval_compare.py 仍加载 LoRA merge 后的模型。")

    append_run(
        {
            "script": "09_full_sft_135m",
            "device": str(device),
            "model": model_dir.name,
            "full_params": total,
            "full_trainable": trainable,
            "trainable_pct": round(pct(trainable, total), 4),
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.lr,
            "train_loss": loss,
            "output": str(OUTPUT_DIR),
        }
    )
    print("完成。记录已写入 reports/week03_runs.jsonl")


def main():
    """入口：解析 --epochs / --max-steps / --lr，固定随机种子后调用 train。"""
    parser = argparse.ArgumentParser(description="135M 全量 SFT（对照 sft001 的 LoRA）")
    parser.add_argument("--epochs", type=int, default=3, help="默认 3；LoRA 版 sft001 用 8")
    parser.add_argument("--max-steps", type=int, default=0, help=">0 时按步数早停，方便冒烟")
    parser.add_argument("--lr", type=float, default=2e-5, help="全量常用比 LoRA 更小的学习率")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    torch.manual_seed(42)
    train(args)


if __name__ == "__main__":
    main()
