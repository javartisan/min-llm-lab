"""Day 1 / 脚本 3：encode 把文字变成数字，decode 把数字变回文字。

学什么
    encode：文本 → input_ids
    decode：input_ids → 文本
    这是 tokenizer 的双向翻译，和 Transformer 里的 Encoder 模块不是一回事。

对应阅读
    docs/学习路线/材料/模型文件说明/02-tokenizer.json结构说明.md 第 3、4 节

运行
    python learn/week01/03_encode_decode.py
"""

from _common import load_tokenizer


def main():
    tokenizer, _ = load_tokenizer()
    text = "Java interface 是什么？"

    print(f"原文: {text}")
    print()

    print("=" * 60)
    print("【1】encode：文字 → 数字")
    print("=" * 60)
    # add_special_tokens=False：先不加 bos/eos，方便和 tokenize 的结果对照
    ids_plain = tokenizer.encode(text, add_special_tokens=False)
    print(f"encode(不加特殊符号) = {ids_plain}")

    # 有的 tokenizer 会在两端加 bos/eos；SmolLM2 对普通 encode 常常不加
    ids_special = tokenizer.encode(text, add_special_tokens=True)
    print(f"encode(加特殊符号)   = {ids_special}")
    if ids_plain == ids_special:
        print("本 tokenizer 对这句话没有自动加 bos/eos。")
        print("对话场景里的特殊符号，主要靠 chat_template 插入（见 10_sft_messages.py）。")
    else:
        print("对比两行：多出来的数字通常就是 bos / eos 这类特殊 token。")

    print()
    print("=" * 60)
    print("【2】decode：数字 → 文字")
    print("=" * 60)
    back_plain = tokenizer.decode(ids_plain)
    back_special = tokenizer.decode(ids_special)
    back_skip = tokenizer.decode(ids_special, skip_special_tokens=True)
    print(f"decode(普通 ids)               = {back_plain!r}")
    print(f"decode(带特殊符号 ids)         = {back_special!r}")
    print(f"decode(带特殊符号, 跳过特殊符) = {back_skip!r}")

    print()
    print("=" * 60)
    print("【3】最容易混的两个词")
    print("=" * 60)
    print("tokenizer.encode  = 文本切成 token，再查成 input_ids")
    print("Transformer Encoder = 神经网络里的「编码器层」（BERT/BART 才有）")
    print("SmolLM2 是 Decoder-only，没有独立 Encoder 模块。")
    print("但问答时一定有 tokenizer.encode。")

    print()
    print("小结论：")
    print("  - encode / decode 是分词器的翻译，不是 Transformer Encoder。")
    print("  - 往返后应能还原原文（特殊符号可选择是否显示）。")


if __name__ == "__main__":
    main()
