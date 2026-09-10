"""Day 3：没有 Encoder，文本如何变成向量？

学什么
    Decoder-only 模型（SmolLM2）没有独立 Encoder 模块。
    但文本照样会变成向量，路径是：

        input_ids（整数）→ Embedding 查表 → 每个 token 一个向量

    SmolLM2-135M 的向量维度 hidden_size = 576。
    Decoder 各层继续在这些向量上做运算，最后才变成 logits / 文字。

对应阅读
    docs/学习路线/材料/Encoder-Decoder与LLM问答流程.md 第 8、9 节

运行
    python learn/week01/08_embedding.py
"""

import torch

from _common import load_model


def main():
    tokenizer, model, model_dir = load_model()
    model.eval()

    text = "Java"
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    print(f"模型目录 : {model_dir}")
    print(f"原文     : {text}")
    print(f"input_ids: {ids.tolist()}   shape={tuple(ids.shape)}")

    # Embedding 就是一张大表：每一行对应词表里的一个 id
    # 形状一般为 [vocab_size, hidden_size] = [49152, 576]
    embedding = model.get_input_embeddings()
    weight = embedding.weight
    print()
    print("=" * 60)
    print("【1】Embedding 是一张查找表，不是 Encoder")
    print("=" * 60)
    print(f"embedding.weight.shape = {tuple(weight.shape)}  # [词表大小, 向量维度]")
    print("含义：词表里每一个 token，都预先存了一条长度为 576 的向量。")

    print()
    print("=" * 60)
    print("【2】把 id 送进 Embedding = 按行查表")
    print("=" * 60)
    with torch.no_grad():
        vectors = embedding(ids)  # [1, seq_len, 576]
    print(f"vectors.shape = {tuple(vectors.shape)}  # [batch, seq_len, hidden_size]")

    # 手工验证：第 0 个 token 的向量，应该等于表里第 id 行
    token_id = int(ids[0, 0])
    from_layer = vectors[0, 0]
    from_table = weight[token_id]
    same = torch.allclose(from_layer, from_table)
    print(f"token_id = {token_id}")
    print(f"手工取 weight[{token_id}] 与 Embedding 层输出是否一致: {same}")
    print(f"这个向量前 8 维: {from_layer[:8].detach().tolist()}")

    print()
    print("=" * 60)
    print("【3】Decoder 层与层之间传的也是向量，不是汉字")
    print("=" * 60)
    print("三种「输出」不要混：")
    print("  Embedding / Decoder 层 → 向量（给下一层算）")
    print("  lm_head                → logits（对词表打分）")
    print("  tokenizer.decode       → 可读文字")
    print()
    print("所以：Decoder 名叫解码器，是因为它负责「生成序列」；")
    print("      内部仍然全程做向量运算，最后才变成字。")

    print()
    print("小结论：")
    print("  - 没有独立 Encoder，也会 Embedding 成向量。")
    print("  - tokenizer.encode ≠ Transformer Encoder。")


if __name__ == "__main__":
    main()
