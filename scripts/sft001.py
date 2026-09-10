"""
sft001.py —— 初学者友好的 SFT（Supervised Fine-Tuning）训练脚本

你将学到：
1. Base 模型 vs Instruct 模型有什么区别
2. 为什么要把数据整理成「对话 messages」格式
3. 什么是 LoRA（只训练少量参数，省内存、少过拟合）
4. SFTConfig 里常见超参数分别控制什么
5. 训练结束后如何保存「可直接加载」的完整模型

运行：
    source venv/bin/activate
    python scripts/sft001.py
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 0) Intel Mac + 旧版 PyTorch 兼容补丁
# ---------------------------------------------------------------------------
# 你的环境是 torch 2.2.2（Intel Mac 上 pip 能装到的最后一版官方轮子）。
# 新版 transformers 检查 bf16 时会调用 torch.backends.mps.is_macos_or_newer，
# 但 2.2.2 里没有这个函数，所以这里补一个兼容实现，避免训练启动就崩溃。
import torch.backends.mps as _mps

if not hasattr(_mps, "is_macos_or_newer"):

    def _is_macos_or_newer(major: int, minor: int = 0) -> bool:
        # 旧 API 只能判断「是否 macOS 13+」
        if major > 13:
            return False
        if major == 13:
            return _mps.is_macos13_or_newer(minor)
        return True

    _mps.is_macos_or_newer = _is_macos_or_newer

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# 1) 路径与模型选择
# ---------------------------------------------------------------------------
# Instruct 模型：已经过「听指令、按对话回答」微调，比纯 Base 更适合问答 SFT。
# Base 模型：只会续写文本，直接拿来做问答容易胡言乱语。
HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

ROOT = Path(__file__).resolve().parent.parent
# 底座模型本地目录（首次运行会从 Hugging Face 下载并缓存到这里）
BASE_DIR = ROOT / "models" / "SmolLM2-135M-Instruct"
# 训练过程中的中间 checkpoint
CHECKPOINT_DIR = ROOT / "checkpoints" / "smollm2-135m-java-lora"
# 最终导出的「合并后完整模型」，评测脚本可以直接加载
OUTPUT_DIR = ROOT / "models" / "smollm2-135m-java-sft"
# 训练数据：每行一个 JSON，字段为 prompt / completion
TRAIN_FILE = ROOT / "data" / "train.jsonl"

# 国内访问 Hugging Face 不稳定时，自动走镜像（也可自行 export HF_ENDPOINT=...）
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# system 提示：告诉模型「你是谁、该怎么回答」
# 小模型很吃这个，能显著减少跑题/瞎续写。
SYSTEM_PROMPT = (
    "你是一个简洁的 Java 助教。请用一两句中文准确回答，不要跑题，不要输出无关代码。"
)


def is_local_model(path: Path) -> bool:
    """判断本地目录是否已经是一个可用的 transformers 模型。"""
    return path.exists() and (path / "config.json").exists()


def load_or_download_base():
    """
    优先从本地加载底座；没有则下载并保存，方便以后离线训练。
    """
    if is_local_model(BASE_DIR):
        print(f"[1/6] 从本地加载 Instruct 底座: {BASE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(BASE_DIR, local_files_only=True)
    else:
        print(f"[1/6] 首次下载 Instruct 底座: {HF_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID)
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(BASE_DIR)
        model.save_pretrained(BASE_DIR)
        print(f"      已保存到: {BASE_DIR}")

    # Causal LM 生成时需要 pad_token；很多模型默认只有 eos_token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False  # 训练时关闭 KV cache，省事也更稳
    return tokenizer, model


def to_messages(example: dict) -> dict:
    """
    把 {prompt, completion} 转成对话 messages。

    为什么要这样做？
    - Instruct 模型训练/推理都习惯 ChatML 对话格式
      （例如 <|im_start|>user ... <|im_end|>）
    - TRL 的 SFTTrainer 看到 messages 字段后，会自动用 tokenizer 的
      chat_template 拼成训练文本，并尽量只对 assistant 部分计算 loss

    messages 结构：
      system  -> 角色设定
      user    -> 用户问题（来自 prompt）
      assistant -> 标准答案（来自 completion）
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]},
        ]
    }


def build_dataset():
    """加载 jsonl，并转换成 messages 格式。"""
    print(f"[2/6] 加载训练数据: {TRAIN_FILE}")
    raw = load_dataset("json", data_files=str(TRAIN_FILE), split="train")
    # map：对每一条样本调用 to_messages
    dataset = raw.map(to_messages, remove_columns=raw.column_names)
    print(f"      样本数: {len(dataset)}")
    print("      示例 messages:")
    for msg in dataset[0]["messages"]:
        print(f"        - {msg['role']}: {msg['content']}")
    return dataset


def build_lora_config() -> LoraConfig:
    """
    LoRA = Low-Rank Adaptation（低秩适配）

    直观理解：
    - 全量微调：更新模型全部参数（慢、占内存、小数据容易过拟合）
    - LoRA：在注意力的线性层旁路加很小的「补丁矩阵」，只训练这些补丁

    常用参数：
    - r: 秩，越大表达能力越强，但也更占显存/更易过拟合。小模型常用 8/16
    - lora_alpha: 缩放系数，实际缩放约等于 alpha/r
    - lora_dropout: 训练时随机丢弃，缓解过拟合
    - target_modules: 把 LoRA 打在哪些层。这里覆盖注意力 + MLP 常见投影
    - task_type: CAUSAL_LM 表示因果语言模型（从左到右生成）
    """
    print("[3/6] 配置 LoRA")
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def build_training_args() -> SFTConfig:
    """
    SFTConfig 继承自 TrainingArguments，是 TRL 做监督微调时的配置中心。

    初学者重点看这些字段：
    - num_train_epochs: 整份数据重复学几轮。数据少可稍大，数据多别太大
    - per_device_train_batch_size: 每次塞进模型的样本数。内存不够就设 1
    - gradient_accumulation_steps: 累积多少小 batch 再更新一次参数
        有效 batch ≈ batch_size * accumulation_steps
    - learning_rate: 学习率。LoRA 常用比全量微调略大一点，如 1e-4
    - logging_steps / save_steps: 隔多少步打印日志 / 存 checkpoint
    - bf16 / fp16: 混合精度。Intel Mac + 旧 torch 建议都关，用 fp32
    - loss_type: 你这套环境用 "nll" 最稳（避免旧 torch 缺 DTensor）
    - packing: 是否把多条短样本拼进同一序列。问答入门建议 False，更直观
    """
    print("[4/6] 配置训练超参数")
    return SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=8,  # 数据还不多，多学几轮；以后数据变多可降到 2~3
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 有效 batch size ≈ 8
        learning_rate=1e-4,  # LoRA 常用学习率
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",  # 不上报 wandb 等平台
        bf16=False,
        fp16=False,
        loss_type="nll",
        packing=False,
        # 助手回复结束后的结束符；有助于模型学会「答完就停」
        eos_token="<|im_end|>",
    )


def train():
    tokenizer, model = load_or_download_base()
    dataset = build_dataset()
    lora_config = build_lora_config()
    training_args = build_training_args()

    print("[5/6] 开始 SFT 训练（LoRA）...")
    # SFTTrainer：TRL 封装好的监督微调训练器
    # - peft_config: 传入后会自动把模型包成 LoRA 模型
    # - processing_class: 新版 TRL 用它代替旧参数 tokenizer
    # - train_dataset: 需要含 messages（或 prompt/completion）字段
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    train_result = trainer.train()
    print(f"      train_loss = {train_result.training_loss:.4f}")

    # -----------------------------------------------------------------------
    # 6) 保存模型
    # -----------------------------------------------------------------------
    # trainer.model 现在是 PeftModel（底座 + LoRA 适配器）。
    # 为了让评测脚本像加载普通模型一样简单，这里把 LoRA 权重 merge 进底座，
    # 导出一份「完整模型」。
    print(f"[6/6] 合并 LoRA 并保存到: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 先保存一份适配器（占用小，方便以后继续训练）
    adapter_dir = CHECKPOINT_DIR / "adapter_final"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"      LoRA 适配器: {adapter_dir}")

    # merge_and_unload：把 LoRA 补丁合并回原权重，并卸掉 peft 包装
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"      合并后完整模型: {OUTPUT_DIR}")
    print("完成！接下来可运行: python scripts/eval_compare.py")


if __name__ == "__main__":
    # 固定随机种子，让结果更容易复现（仍无法做到绝对一致）
    torch.manual_seed(42)
    train()
