"""Day 1 / 脚本 4：换几句中英文，观察中文为何常被拆碎。

学什么
    SmolLM2 的词表是通用 BPE，不是按汉字一个字一个 id 建的。
    英文常见词往往整块是一个 token；中文/少见组合更容易被拆成多个子词。

对应阅读
    docs/学习路线/材料/模型文件说明/02-tokenizer.json结构说明.md 第 1、5 节
    学习计划第 1 周 Day 1：换 3 句中英文，观察中文为何常被拆碎

运行
    python learn/week01/04_chinese_vs_english.py

----------------------------------------------------------------------
科普 1：词表不是「汉字字典」

模型不算汉字，只算整数。词表是一张固定对照表：

    某个碎片  →  一个 id
    "Java"    →  39570     （本 tokenizer 上常为 1 个 token）
    "什么"    →  不在整词表里，会被拆成多个 id

SmolLM2 词表大约 49152 行，预训练时就定死了。
本项目做 Java 问答 SFT，一般不会按中文/Java 再编一套码。
所以 tokenizer.json ≠ Java 领域字符表。

----------------------------------------------------------------------
科普 2：「一个汉字一个 id」vs 现在实际用的 BPE

如果按汉字字典建词表，理想情况是：

    你 → 一个 id，好 → 一个 id，「什么」两个字两个 id（或整词一个 id）

优点：中文一眼能对上。
缺点：汉字、英文、代码、标点全要占坑，词表极大；没见过的组合仍要拆。

SmolLM2 不是这种表。常见汉字的「整字字符串」往往不在词表里。

----------------------------------------------------------------------
科普 3：BPE 在干什么（Byte Pair Encoding，字节对编码）

可以想成：从很小的片开始，把经常挨在一起的两片黏成更大的一块。

    1. 先拆到很小。SmolLM2 是 Byte-level BPE：先变成 UTF-8 字节。
    2. 统计哪两片最常连着出现，合并成新 token，写进词表。
    3. 重复很多次，词表涨到约 49152 就停。

英文里 the / interface / Java 出现极多，所以常常整词就是 1 个 token。
中文在这份通用语料里「整词高频」更少，很多字会先落到 UTF-8 的几个字节上：

    什么  →  往往 4 个 token（看起来像 ä» Ģ ä¹ Ī）
    多态  →  往往 4 个 token
    你    →  往往 2 个 token

那些 ä»、Ġ 看起来像乱码，只是字节/子词的显示方式。
整段 decode 回去仍是正确中文。

Ġ 在 Byte-level BPE 里通常表示「前面有个空格」，
所以 token 'Ġthe' 其实是带空格的单词 the。

「通用」= 这张表给各种语言、代码、网页共用，不是 Java 关键字表，
也不是中文 2 万汉字表。领域知识在模型权重里，不在词表里。

----------------------------------------------------------------------
科普 4：tokenize 和 encode 差在哪（本脚本两个都会用到）

    tokenize(text)  →  文字碎片列表（给人看怎么切的）
    encode(text)    →  input_ids 整数列表（给模型算）

关系：encode ≈ tokenize + convert_tokens_to_ids
（encode 还可以选择是否加 bos/eos；本脚本用 tokenize 方便对照碎片。）

----------------------------------------------------------------------
科普 5：除了 BPE 还有哪些切法

切词可以按「切得多碎」排成一条光谱：

    整词（很粗） → 子词（BPE 等） → 单字 → 字节（最细）

    整词 Word-level
        按空格/词典切开。生词容易变成 UNK；中文还得先分词。现代 LLM 几乎不用。

    单字 Character-level
        英文一个字母一个 id；中文可以一个汉字一个 id。
        词表小、无未知词，但序列很长。现在 LLM 主流不是这条。

    子词（现在的主流）
        BPE       反复合并最常见的相邻两片。GPT-2、SmolLM2 等。
        WordPiece 也合并，计分规则不同。BERT 常用。
        Unigram   先一大堆候选再删不太有用的。常和 SentencePiece 一起（T5 等）。
        SentencePiece 不是第四种算法，而是一套实现（可跑 BPE 或 Unigram）。

    纯字节
        每个字节 0–255 一个 token。永不未知，但中文更长。ByT5 这类会用。
        SmolLM2 是「字节打底 + BPE 合并」，不是纯 256 词表。

工业界搜索/传统中文 NLP 常用结巴分词，那是另一路；
送进 SmolLM 之前，还是会再走它自己的 BPE。

对微调的影响：中文 token 往往比英文多（同样一句话序列更长）；
单个 token 不像一个汉字是正常的；看 decode 后的整句即可。
"""

from _common import load_tokenizer


def show(tokenizer, title: str, text: str):
    """打印一句话被切成了多少片。

    这里用 tokenize 而不是 encode，是为了同时看到「碎片长什么样」。
    真正进模型的仍是下面的 input_ids（和 encode 的结果通常一致）。
    """
    # tokenize：停在文字碎片；encode：直接得到数字
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\n[{title}]")
    print(f"  原文      : {text}")
    print(f"  token 数  : {len(tokens)}")
    print(f"  tokens    : {tokens}")
    print(f"  input_ids : {ids}")
    # Byte-level BPE 的单个 token 可能像乱码（UTF-8 字节的显示），整段 decode 会恢复
    print(f"  拼回去    : {tokenizer.decode(ids)!r}")


def main():
    tokenizer, _ = load_tokenizer()

    print("同一个 tokenizer，中文句子往往 token 更多。")
    print("原因：通用 Byte-level BPE，不是「一个汉字 = 一个 id」。")
    print("英文高频词常整块 1 个 token；中文常被拆成多个字节/子词。")

    # 计划要求：换 3 句中英文对照（尽量意思接近，方便数 token）
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
    print("英文 3 句（常见词更容易整块留下）")
    print("=" * 60)
    for i, text in enumerate(english_list, start=1):
        show(tokenizer, f"EN-{i}", text)

    print("\n" + "=" * 60)
    print("中文 3 句（整词常不在词表，会被拆碎）")
    print("=" * 60)
    for i, text in enumerate(chinese_list, start=1):
        show(tokenizer, f"ZH-{i}", text)

    print("\n小结论：")
    print("  - 词表是预训练定好的通用碎片表，不是汉字字典，也不是 Java 字典。")
    print("  - BPE：高频相邻片合并。英文常见词常 1 片；中文常多片。")
    print("  - 单个 token 不像汉字没关系，decode 后仍是正确中文。")
    print("  - 子词还有 WordPiece / Unigram；更粗有整词，更细有单字/纯字节。")
    print("  - 模型看的是 id 序列；微调通常改权重，不改这张词表。")


if __name__ == "__main__":
    main()
