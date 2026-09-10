"""Day 1 / 脚本 5：特殊 token 和词表查询。

学什么
    特殊 token 不是普通汉字/单词，而是给模型看的「控制符号」。
    Instruct 对话特别依赖 <|im_start|> / <|im_end|>。
    词表就是「字符串 → id」的大字典。

对应阅读
    docs/学习路线/材料/模型文件说明/02-tokenizer.json结构说明.md 第 2.3、2.8 节

运行
    python learn/week01/05_special_tokens.py
"""

from _common import load_tokenizer


def main():
    tokenizer, _ = load_tokenizer()

    print("=" * 60)
    print("【1】对话常用特殊符号")
    print("=" * 60)
    names = [
        "<|im_start|>",  # 一段对话角色开始，后面跟 system/user/assistant
        "<|im_end|>",  # 这一段说完了
        "<|endoftext|>",  # 更通用的结束/未知相关符号
    ]
    for name in names:
        # convert_tokens_to_ids：查这个字符串在词表里的编号
        tid = tokenizer.convert_tokens_to_ids(name)
        print(f"  {name:<16} → id = {tid}")

    print()
    print("bos/eos/pad 实际绑定到了哪个符号：")
    print(f"  bos = {tokenizer.bos_token!r}  (id={tokenizer.bos_token_id})")
    print(f"  eos = {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    print(f"  pad = {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")

    print()
    print("=" * 60)
    print("【2】从词表里查一个普通词")
    print("=" * 60)
    # tokenizer.get_vocab() 返回 dict：token字符串 → id
    vocab = tokenizer.get_vocab()
    print(f"词表条目数 = {len(vocab)}")
    print(f"词表条类型 = {type(vocab)}")

    # 英文常见词通常能直接查到；中文整词不一定在词表里
    for piece in ["Java", "interface", "什么", "多态"]:
        if piece in vocab:
            print(f"  {piece!r} 在词表中，id={vocab[piece]}")
        else:
            # 不在词表，就会被 BPE 继续拆成更小的片
            parts = tokenizer.tokenize(piece)
            print(f"  {piece!r} 不在词表整词中，会被拆成 {parts}")

    print()
    print("小结论：")
    print("  - 特殊 token 用来标记对话结构（谁在说话、这句话结束了）。")
    print("  - 词表里没有的字/词，BPE 会拆开再编码，不会直接报错。")


if __name__ == "__main__":
    main()
