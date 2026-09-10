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

----------------------------------------------------------------------
科普 1：大模型里有没有内嵌 Embedding？

有，但是「内部的一层」，不是外面再挂一个独立的 Embedding 模型。

    文本
      → Tokenizer（在模型外面）→ input_ids
      → Embedding 层（在模型最前面）→ 向量
      → Decoder 各层 → 向量
      → lm_head → logits → 选下一个 token

Tokenizer 负责文字 ↔ 数字；Embedding 负责数字 → 向量。

----------------------------------------------------------------------
科普 2：get_input_embeddings() 拿到的是什么？

    embedding = model.get_input_embeddings()

对本仓库 SmolLM2（LlamaForCausalLM）实测：

    类型     nn.Embedding
    就是     model.model.embed_tokens
    权重形状 [49152, 576]   # vocab_size × hidden_size

它是一张查找表：每一行对应词表里一个 token id 的向量。
embedding(ids) 等价于按行取 weight[id]，不是再跑一个完整小模型。

----------------------------------------------------------------------
科普 3：别和「Embedding 模型」混为一谈

    大模型里的 Embedding 层          独立 Embedding 模型（如 BGE / E5）
    LLM 内部的一层                   另一个完整模型
    每个 token 一个向量 [seq, 576]   通常一句一个向量，做检索/相似度
    get_input_embeddings() 取出      另外 from_pretrained 加载

可以说「大模型内嵌了 embedding」，
不要理解成「里面还藏着一个 sentence-transformers」。

----------------------------------------------------------------------
科普 4：输出那边也有一层，常常和输入共用权重

    get_output_embeddings()  →  lm_head（隐向量映回词表打分）

SmolLM2 配置 tie_word_embeddings=True，输入表和输出表共用同一份权重：

    输入：id → 向量（Embedding）
    输出：向量 → 词表分数（lm_head）
    两边往往是同一张表转置着用
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

    # 取出模型内部的输入 Embedding 层（查表），不是独立的检索模型
    embedding = model.get_input_embeddings()
    weight = embedding.weight
    print()
    print("=" * 60)
    print("【1】Embedding 是一张查找表，不是 Encoder，也不是独立 Embedding 模型")
    print("=" * 60)
    print(f"类型                   : {type(embedding).__name__}")
    print(f"是否即 model.model.embed_tokens : {embedding is model.model.embed_tokens}")
    print(f"embedding.weight.shape = {tuple(weight.shape)}  # [词表大小, 向量维度]")
    print("含义：词表里每一个 token，都预先存了一条长度为 576 的向量。")

    print()
    print("=" * 60)
    print("【2】把 id 送进 Embedding = 按行查表")
    print("=" * 60)
    with torch.no_grad():
        # vectors = embedding(ids) 含义：用 token 编号当行号，去 Embedding 表里取出向量。
        #
        # embedding 是 nn.Embedding 对象，embedding(ids) 调用的是它的 __call__，
        # 不是再跑一个完整小模型，只是按行查表。
        #
        #   ids               形状 [batch, seq_len]，里面是整数编号
        #                     例如 "Java" → [[39570]]
        #   embedding.weight  形状 [49152, 576]，第 k 行 = id=k 的 576 维向量
        #   vectors           形状 [batch, seq_len, 576]
        #                     对本句即 weight[39570]，也就是 [1, 1, 576]
        #
        # 若句子有 11 个 token，就会查出 11 行，拼成 [1, 11, 576]。
        # 下面 from_layer == from_table 就是在验证：查表层 == 手工取 weight[id]。
        vectors = embedding(ids)  # [1, seq_len, 576]
    print(f"vectors.shape = {tuple(vectors.shape)}  # [batch, seq_len, hidden_size]")

    # 手工验证：第 0 个 token 的向量，应该等于表里第 id 行
    # ids 形状 [batch, seq_len]，这里 batch=0、位置=0，即这句话的第一个 token
    token_id = int(ids[0, 0])
    # vectors 形状 [batch, seq_len, 576]：Embedding 层查表后、该位置上的那条向量
    from_layer = vectors[0, 0]
    # weight 形状 [vocab_size, 576]：不经过层，直接用编号当行号取出同一行
    from_table = weight[token_id]
    # allclose：两个浮点向量是否几乎相等（允许极小误差）。True 说明「调用层 = 按行取表」
    same = torch.allclose(from_layer, from_table)
    print(f"token_id = {token_id}")
    print(f"手工取 weight[{token_id}] 与 Embedding 层输出是否一致: {same}")
    print(f"这个向量前 8 维: {from_layer[:8].detach().tolist()}")

    print()
    print("=" * 60)
    print("【3】输出层 lm_head 常常和输入 Embedding 绑在同一张表上")
    print("=" * 60)
    lm_head = model.get_output_embeddings()
    tied = model.config.tie_word_embeddings
    same_ptr = lm_head.weight.data_ptr() == embedding.weight.data_ptr()
    print(f"get_output_embeddings 类型 : {type(lm_head).__name__}")
    print(f"tie_word_embeddings        : {tied}")
    print(f"lm_head 与 Embedding 共用权重: {same_ptr}")
    print("输入：id → 向量；输出：向量 → 词表分数。绑定时两边是同一份参数。")

    print()
    print("=" * 60)
    print("【4】Decoder 层与层之间传的也是向量，不是汉字")
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
    print("  - get_input_embeddings() 取的是内部查表层，不是 BGE 那种检索模型。")
    print("  - SmolLM2 常把输入 Embedding 和 lm_head 绑成同一张表。")


if __name__ == "__main__":
    main()
