"""打印文本对应的 token（文字片段）和 input_ids（数字编号）。

运行：
    python scripts/print_tokens.py
    python scripts/print_tokens.py "什么是 Java 多态？"
"""

from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

# 优先用本地已有的 tokenizer（Instruct / Base / 微调模型均可）
CANDIDATES = [
    ROOT / "models" / "SmolLM2-135M-Instruct",
    ROOT / "models" / "SmolLM2-135M",
    ROOT / "models" / "smollm2-135m-java",
]


def find_tokenizer_dir() -> Path:
    for path in CANDIDATES:
        if (path / "tokenizer_config.json").exists() or (path / "tokenizer.json").exists():
            return path
    raise FileNotFoundError(
        "本地未找到 tokenizer，请先运行 python scripts/sft001.py 或 python scripts/main.py 下载模型。"
    )


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "Java interface 是什么？"
    model_dir = find_tokenizer_dir()
    print(f"Tokenizer: {model_dir}")
    print(f"原文: {text}")
    print("-" * 50)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # token：文本片段（人能看懂的字符串）
    tokens = tokenizer.tokenize(text)
    # input_ids：每个 token 对应的整数编号
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"{'序号':<6}{'token(原始)':<20}{'可读文字':<16}{'input_id':<10}")
    print("-" * 60)
    for i, (tok, tid) in enumerate(zip(tokens, input_ids)):
        raw = tok.replace("Ġ", "▁")
        readable = tokenizer.decode([tid])
        print(f"{i:<6}{raw:<20}{readable:<16}{tid:<10}")

    print("-" * 60)
    print(f"tokens     = {tokens}")
    print(f"input_ids  = {input_ids}")
    print(f"拼回原文   = {tokenizer.decode(input_ids)}")
    print(f"token 数量 = {len(tokens)}")

    # 再用 encode 验证：结果应与上面 input_ids 一致（是否加特殊符号取决于 tokenizer）
    encoded = tokenizer.encode(text, add_special_tokens=False)
    print(f"encode()   = {encoded}")


if __name__ == "__main__":
    main()
