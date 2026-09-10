"""本周脚本共用的「找文件 / 加载模型」小工具。

学习脚本都尽量短，所以把重复的路径判断集中放在这里。
你暂时不必深究这个文件，只要知道：

- 仓库根目录叫 ROOT
- 训练数据在 data/train.jsonl、data/test.jsonl
- tokenizer / 模型优先用本地 models/ 目录，避免重复下载
"""

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# learn/week01/ 再上两级才是仓库根目录
ROOT = Path(__file__).resolve().parent.parent.parent

# 训练 / 测试数据（第 1 周 Day 2 会用到）
TRAIN_FILE = ROOT / "data" / "train.jsonl"
TEST_FILE = ROOT / "data" / "test.jsonl"

# 优先用 Instruct：第 1 周后面会学「问答 SFT 为什么用 Instruct」
# 没有 Instruct 时再退回 Base / 已微调模型
_MODEL_CANDIDATES = [
    ROOT / "models" / "SmolLM2-135M-Instruct",
    ROOT / "models" / "SmolLM2-135M",
    ROOT / "models" / "smollm2-135m-java-sft",
    ROOT / "models" / "smollm2-135m-java",
]


def find_model_dir() -> Path:
    """找到本地已经下载好的模型目录（里面要有 config.json）。"""
    for path in _MODEL_CANDIDATES:
        if (path / "config.json").exists():
            return path
    raise FileNotFoundError(
        "本地未找到模型。请先运行 python scripts/main.py 或 python scripts/sft001.py 下载。"
    )


def load_tokenizer():
    """只加载分词器，不加载模型权重（Day 1 / Day 2 用，启动更快）。"""
    model_dir = find_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return tokenizer, model_dir


def load_model():
    """加载分词器 + 完整模型（Day 3 之后的前向 / 损失需要权重）。"""
    model_dir = find_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

    # 很多因果语言模型默认没有 pad_token，生成时需要一个填充符。
    # 常见做法：用 eos_token 兼任 pad_token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 第 1 周在 Intel Mac 上用 CPU 即可，先把数据流跑明白。
    model.to("cpu")
    return tokenizer, model, model_dir
