"""Day 1 / 脚本 2：看清 token 和 input_id 的区别。

学什么
    token     = 文字片段（人勉强能看懂）
    input_id  = 这个片段在词表里的整数编号（模型真正吃进去的东西）

对应阅读
    docs/学习路线/材料/模型文件说明/02-tokenizer.json结构说明.md 第 5 节

运行
    python learn/week01/02_token_and_id.py
    python learn/week01/02_token_and_id.py "什么是 Java 多态？"
"""

import sys

from _common import load_tokenizer


def main():
    # 允许命令行传入自己的句子，方便多试几句
    text = sys.argv[1] if len(sys.argv) > 1 else "how are you？"

    tokenizer, model_dir = load_tokenizer()
    print(f"Tokenizer: {model_dir}")
    print(f"原文: {text}")
    print("-" * 60)

    # tokenize：只切成片段，还没有变成数字
    tokens = tokenizer.tokenize(text)
    # convert_tokens_to_ids：拿每个片段去词表里查编号
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"{'序号':<6}{'token(原始)':<22}{'decode后':<16}{'input_id':<10}")
    print("-" * 60)
    for i, (tok, tid) in enumerate(zip(tokens, input_ids)):
        # Byte-level BPE 常用 Ġ 表示「前面有个空格」。打印时换成 ▁ 更好认。
        raw = tok.replace("Ġ", "▁")
        # 单个 id decode 回去，方便对照「这个数字对应哪段文字」
        readable = tokenizer.decode([tid])
        print(f"{i:<6}{raw:<22}{readable:<16}{tid:<10}")

    print("-" * 60)
    print(f"tokens     = {tokens}")
    print(f"input_ids  = {input_ids}")
    print(f"token 数量 = {len(tokens)}")

    print()
    print("小结论：")
    print("  - token 是片段，input_id 是数字。模型计算用的是数字。")
    print("  - 同一个字/词每次编码得到的 id 是固定的（词表不变）。")


if __name__ == "__main__":
    main()
