"""Day 4：Loss + Backward + 参数更新。

学什么
    训练一步的后半段：

        logits → Loss（和标签比误差）→ Backward（算梯度）→ 优化器改权重

    Loss 越小，通常表示预测越贴近标签。
    Backward 不算新文本，只算「每个参数该往哪边改」。

对应阅读
    docs/学习路线/材料/微调学习笔记.md 第 5 节

运行
    python learn/week01/09_loss_backward.py
"""

import torch

from _common import load_model


def main():
    tokenizer, model, model_dir = load_model()
    print(f"模型目录: {model_dir}")

    # ---------------------------------------------------------------------
    # 准备一条极短文本，只为把「一步训练」看清楚（不是认真微调）
    # ---------------------------------------------------------------------
    text = "Java interface"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"文本      : {text}")
    print(f"input_ids : {input_ids.tolist()}")

    # 因果语言模型的标准做法：labels 就是 input_ids
    # 模型内部会右移：用位置 i 的隐藏向量，去预测位置 i+1 的 token
    labels = input_ids.clone()

    # ---------------------------------------------------------------------
    # 为了演示「有的参数会动」，这里只解冻 Embedding（其它层冻住）
    # 真实 SFT 会按全量或 LoRA 来选哪些参数可训练，原理相同
    # ---------------------------------------------------------------------
    for p in model.parameters():
        p.requires_grad = False
    embedding = model.get_input_embeddings()
    embedding.weight.requires_grad_(True)

    # SGD 最好懂：新参数 = 旧参数 - 学习率 × 梯度
    optimizer = torch.optim.SGD(embedding.parameters(), lr=1e-2)

    token_id = int(input_ids[0, 0])
    before = embedding.weight[token_id, 0].item()

    print()
    print("=" * 60)
    print("【1】Forward + Loss")
    print("=" * 60)
    model.train()  # 训练模式：需要梯度
    optimizer.zero_grad()  # 先清空上一次残留梯度
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    print(f"loss = {loss.item():.4f}")
    print("这个数是「平均负对数似然」。越大 = 越不像标签；训练就是把它压下去。")

    print()
    print("=" * 60)
    print("【2】Backward：反传，算出梯度")
    print("=" * 60)
    loss.backward()
    grad = embedding.weight.grad
    print(f"Embedding.grad 是否存在 : {grad is not None}")
    print(f"Embedding.grad 形状     : {tuple(grad.shape)}")
    print(f"梯度的整体大小(L2)      : {grad.norm().item():.6f}")
    print("梯度告诉优化器：每个参数往哪边改、改多少。")
    print("这一步没有产生新的回答文本。")

    print()
    print("=" * 60)
    print("【3】参数更新")
    print("=" * 60)
    optimizer.step()
    after = embedding.weight[token_id, 0].item()
    print(f"某个权重更新前 = {before:.8f}")
    print(f"某个权重更新后 = {after:.8f}")
    print(f"差值           = {after - before:.8f}")
    print("公式直觉：weight ← weight - lr * grad")

    print()
    print("和小项目正式训练的关系：")
    print("  - scripts/sft001.py 里的 SFTTrainer 每一步也是 Forward → Loss → Backward → 更新")
    print("  - 差别只是：数据是对话、可训练参数可能是 LoRA、优化器一般是 AdamW")

    print()
    print("小结论：")
    print("  - Loss 衡量「预测和标签差多远」。")
    print("  - Backward 算的是参数怎么改，不是再读一遍新句子。")


if __name__ == "__main__":
    main()
