"""Day 1 / 脚本 1：加载 Tokenizer，认识词表。

学什么
    Tokenizer 不是模型本身，它是「文本 ↔ 数字」的翻译官。
    词表大小 vocab_size 是固定的（SmolLM2 约 49152），微调通常不会重做词表。

对应阅读
    docs/学习路线/材料/模型文件说明/02-tokenizer.json结构说明.md 第 1、2 节

运行
    python learn/week01/01_load_tokenizer.py
"""

from _common import load_tokenizer


def main():
    # AutoTokenizer.from_pretrained 会读取模型目录里的：
    #   tokenizer.json / tokenizer_config.json / special_tokens_map.json 等
    tokenizer, model_dir = load_tokenizer()

    print("=" * 60)
    print("【1】Tokenizer 从哪个目录加载？")
    print("=" * 60)
    print(f"目录: {model_dir}")
    print("提示: tokenizer.json 是通用分词配置，不是 Java 专用字典。")

    print()
    print("=" * 60)
    print("【2】词表有多大？")
    print("=" * 60)
    # len(tokenizer) = 词表里有多少个 token（含特殊符号）
    print(f"len(tokenizer)     = {len(tokenizer)}")
    # vocab_size 来自 tokenizer 配置，一般应和上面接近或相同
    print(f"tokenizer.vocab_size = {tokenizer.vocab_size}")
    print("模型只认识 0 ~ vocab_size-1 这些整数，不认识原始汉字/英文。")

    print()
    print("=" * 60)
    print("【3】几个重要特殊符号（先混个脸熟，05 脚本会细看）")
    print("=" * 60)
    print(f"bos_token (开始) = {tokenizer.bos_token!r}  id={tokenizer.bos_token_id}")
    print(f"eos_token (结束) = {tokenizer.eos_token!r}  id={tokenizer.eos_token_id}")
    print(f"pad_token (填充) = {tokenizer.pad_token!r}  id={tokenizer.pad_token_id}")
    print(f"unk_token (未知) = {tokenizer.unk_token!r}  id={tokenizer.unk_token_id}")

    print()
    print("小结论：")
    print("  - Tokenizer 负责切词和查号，不负责「理解句子」。")
    print("  - 词表是预训练时定好的，本项目微调一般不改词表。")


if __name__ == "__main__":
    main()
