"""Day 1 / 脚本 4：换几句中英文，观察中文为何常被拆碎。

学什么
    SmolLM2 的词表是通用 BPE，不是按汉字一个字一个 id 建的。
    英文常见词往往整块是一个 token；中文/少见组合更容易被拆成多个子词。

对应阅读
    docs/学习路线/材料/模型文件说明/02-tokenizer.json结构说明.md 第 1、5 节
    学习计划第 1 周 Day 1：换 3 句中英文，观察中文为何常被拆碎

运行
    python learn/week01/04_chinese_vs_english.py
"""

from _common import load_tokenizer


def show(tokenizer, title: str, text: str):
    """打印一句话被切成了多少片。"""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\n[{title}]")
    print(f"  原文      : {text}")
    print(f"  token 数  : {len(tokens)}")
    print(f"  tokens    : {tokens}")
    print(f"  input_ids : {ids}")
    # 单个 token 看起来可能像乱码，整段 decode 会恢复正常
    print(f"  拼回去    : {tokenizer.decode(ids)!r}")


def main():
    tokenizer, _ = load_tokenizer()

    print("同一个 tokenizer，中文句子往往 token 更多。")
    print("原因：词表偏英文/多语言子词，不是「一个汉字 = 一个 id」。")

    # 计划要求：换 3 句中英文对照
    english_list = [
        "What is a Java interface?",
        "Can we instantiate an interface?",
        "Polymorphism lets one behavior have many implementations.",
    ]
    chinese_list = [
        "什么是 Java interface？",
        "接口可以实例化吗？",
        "多态让同一行为在不同对象上有不同实现。",
    ]

    print("\n" + "=" * 60)
    print("英文 3 句")
    print("=" * 60)
    for i, text in enumerate(english_list, start=1):
        show(tokenizer, f"EN-{i}", text)

    print("\n" + "=" * 60)
    print("中文 3 句")
    print("=" * 60)
    for i, text in enumerate(chinese_list, start=1):
        show(tokenizer, f"ZH-{i}", text)

    print("\n小结论：")
    print("  - 中文常被拆碎，单个 token 不一定等于一个汉字。")
    print("  - 这不影响训练：模型看的是 id 序列，decode 后仍是正确中文。")
    print("  - tokenizer.json 不是 Java 领域字符表，所以 Java 中文问答也会被拆。")


if __name__ == "__main__":
    main()
