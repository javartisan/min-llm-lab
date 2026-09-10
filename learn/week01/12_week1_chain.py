"""Day 7：把第 1 周知识链串起来（验收用）。

本周要能讲清的 8 个框：

    文本 → Tokenizer → input_ids → Embedding+Decoder → logits
         → Loss → Backward → 参数更新

以及：
    tokenizer.encode ≠ Transformer Encoder
    SmolLM 问答没有独立 Encoder，但会 Embedding 成向量
    Decoder 层间传向量，最后才变成文字

运行
    python learn/week01/12_week1_chain.py
"""

import json

import torch

from _common import TRAIN_FILE, load_model


def main():
    tokenizer, model, model_dir = load_model()
    print(f"使用模型: {model_dir}")
    print()

    with TRAIN_FILE.open(encoding="utf-8") as f:
        sample = json.loads(f.readline())
    text = sample["prompt"]
    print("一条真实训练问题:", text)

    print()
    print("① 文本")
    print("   人能看懂的字符串，模型还不能直接算。")

    print()
    print("② Tokenizer")
    tokens = tokenizer.tokenize(text)
    print(f"   tokens = {tokens}")

    print()
    print("③ input_ids")
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    print(f"   input_ids = {ids.tolist()}")
    print("   这是 tokenizer.encode，不是 Transformer Encoder。")

    print()
    print("④ Embedding + Decoder（没有独立 Encoder）")
    embedding = model.get_input_embeddings()
    with torch.no_grad():
        vectors = embedding(ids)
    print(f"   Embedding 后向量形状 = {tuple(vectors.shape)}  # [1, seq, 576]")
    print("   随后 Decoder 各层继续：向量 → 向量（不是直接出汉字）")

    print()
    print("⑤ logits")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    print(f"   logits.shape = {tuple(logits.shape)}  # [1, seq, vocab]")
    print("   分数还不是概率，更不是完整答案。")

    print()
    print("⑥⑦⑧ Loss → Backward → 更新（训练才有；推理没有）")
    # 先冻住大部分参数，只留 Embedding 可训练，反传才不会又慢又占内存
    for p in model.parameters():
        p.requires_grad = False
    embedding.weight.requires_grad_(True)

    labels = ids.clone()
    model.train()
    optimizer = torch.optim.SGD(embedding.parameters(), lr=1e-2)
    optimizer.zero_grad()
    outputs = model(input_ids=ids, labels=labels)
    print(f"   loss = {outputs.loss.item():.4f}")

    outputs.loss.backward()
    print(f"   Embedding 梯度范数 = {embedding.weight.grad.norm().item():.6f}")
    print("   Backward 没有生成新文本，只算出参数怎么改。")
    optimizer.step()
    print("   optimizer.step() 之后，权重已经往降低 Loss 的方向挪了一点。")

    print()
    print("=" * 60)
    print("验收口答（对照学习计划第 1 周）")
    print("=" * 60)
    print("Q: 问答有没有 Encoder？")
    print("A: 没有独立 Encoder 模块；但一定有 tokenizer.encode 和 Embedding。")
    print()
    print("Q: Decoder 为何产出向量？")
    print("A: Decoder 是神经网络，层与层之间传向量；")
    print("   tokenizer.decode 才把最后的 token id 还原成文字。")
    print()
    print("Q: 本项目 SFT 数据格式？")
    print("A: jsonl 的 prompt/completion，训练前转成 messages，")
    print("   再用 chat_template 拼成对话文本。")
    print()
    print("完整链：Tokenizer → Dataset → Forward → Loss → Backward")


if __name__ == "__main__":
    main()
