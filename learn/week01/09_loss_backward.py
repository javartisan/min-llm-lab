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

----------------------------------------------------------------------
科普：model.train() 是不是「开始训练」？

不是。它不会启动训练，只是把模型切到「训练模式」。

真正改权重的是后面这几步：

    model.train()          ← 只切换开关（Dropout 等按训练行为工作）
    optimizer.zero_grad()  ← 清掉旧梯度
    model(...)             ← Forward，算出 logits / loss
    loss.backward()        ← 反传，算出梯度
    optimizer.step()       ← 真正改参数

train() 和 eval() 差在哪：

    model.train()  训练模式：Dropout 会随机丢、BatchNorm 用当前 batch 统计
                   不会更新权重，只改层的行为
    model.eval()   推理模式：Dropout 关闭、BN 用运行均值
                   同样不会更新权重

07 / 08 用 eval()：只要前向，不要训练行为。
09 用 train()：这一步按「训练时」的规则走。
（SmolLM2 的 dropout 常常是 0，习惯上训练前仍切到 train。）

和「要不要算梯度」也不是一回事：

    能不能反传：有没有包在 torch.no_grad() 里，以及 requires_grad 是否为 True
    会不会改权重：有没有 optimizer.step()

07 里 with torch.no_grad() 是关掉梯度；
09 没有这层，再 backward() + step()，权重才会动。

scripts/sft001.py 的 trainer.train() 才是「启动整段训练循环」
（TRL 封装很多步 Forward → Loss → Backward → 更新）。
单个 model.train() 只相当于告诉模块：现在按训练状态工作。

记一句：model.train() 是换挡，不是踩油门；
油门是 Forward + Loss + Backward + optimizer.step()。

----------------------------------------------------------------------
科普：Dropout 蒙住一部分，推理时没蒙同一处，会不会误差很大？

Dropout 可以想成：训练时临时蒙住一部分神经元，逼模型别死记某一条路。
数字示例见 learn/py/dropout.py（5 个数、p=0.5，不用加载大模型）。

例子里的 2、3 不是句子的第 2、3 个字，而是长度为 5 的向量下标：
某次训练可能把下标 1、2 变成 0。下次前向随机抽的是另一些下标。

推理时不会再随机蒙。eval() 下 Dropout 关掉，每次用全部神经元。
所以不存在「训练蒙了 2、3，推理也必须蒙 2、3」。

训练：随机走很多条残缺的路，学会冗余。
推理：走完整的路，结果更稳。

为了让两边平均起来差不多，PyTorch 用 inverted dropout：
训练时留下的值乘 1/(1-p)。p=0.5 就是乘 2。
推理时不丢、也不乘，期望和训练时对齐。

什么时候误差会真的变大：
    训练 train()、推理 eval()     正确，这就是 Dropout 的用法
    推理还用 train()             每次随机蒙的位置不同，预测会乱跳
    dropout 太大且数据很少       那是超参问题，不是「没蒙对 2、3」

对本仓库 SmolLM2：attention_dropout=0.0，等于规定可以蒙但一张都不蒙，
train/eval 在 Dropout 上几乎没数值差别；习惯上仍要成对使用。

----------------------------------------------------------------------
科普：loss.backward() 和 optimizer.step() 各自干什么？

这两步紧挨着，职责完全不同：backward 只算该往哪改，step 才真的改。

    optimizer.zero_grad()   先把旧梯度清零
    model(...)              Forward，得到 loss
    loss.backward()         反传：算出每个参数的梯度，写进 param.grad
    optimizer.step()        按「学习率 × 梯度」去改 param.data

loss.backward() 完成什么
    从 loss 这个标量沿计算图往回走，用链式法则问：
    每个可训练参数对当前 loss 的贡献是正是负、有多大？
    结果写在 参数.grad 里，例如 embedding.weight.grad。
    会做：算出梯度；把梯度累加到 .grad（所以下一步前通常要 zero_grad）。
    不会：改权重；生成新文本；再读一遍句子。
    比喻：阅卷，标出「这里该加、那里该减」，卷面分数（权重）还没改。

optimizer.step() 完成什么
    拿 .grad 里的数，按优化器公式更新权重。
    本脚本 SGD：weight ← weight - lr × grad
    AdamW 更复杂（动量、自适应学习率），本质一样：用梯度改参数。
    会做：真正改 param.data（所以 before / after 会不同）。
    不会：重新算一遍梯度；默认也不自动清 .grad。
    比喻：按阅卷意见改作业。

为什么必须两步、顺序不能反
    只有 backward()     .grad 有了，权重原地不动
    只有 step()         .grad 还是 None 或上一次的，乱改或报错
    先 step 再 backward  用的不是这一轮的梯度
    两次 backward 不 zero_grad  梯度会累加，步子可能过大

07 用 torch.no_grad()，图都没建，不能 backward。
09 解冻了 Embedding，backward 后 embedding.weight.grad 才不是 None，step 才改得动。

记一句：backward = 算出怎么改（写梯度）；step = 按这个意见改权重。
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
    # 只切换到训练模式（Dropout 等按训练行为），并不会开始改权重
    model.train()
    optimizer.zero_grad()  # 先清空上一次残留梯度
    # 这一次前向会算 loss；没有包在 torch.no_grad() 里，后面才能 backward
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    print(f"loss = {loss.item():.4f}")
    print("这个数是「平均负对数似然」。越大 = 越不像标签；训练就是把它压下去。")
    print("到这里权重还没改。model.train() 也没有触发训练循环。")

    print()
    print("=" * 60)
    print("【2】Backward：反传，算出梯度")
    print("=" * 60)
    # 只写 embedding.weight.grad，不改权重。比喻：阅卷，还没改分数。
    # 梯度会累加，所以前面必须 zero_grad()。
    loss.backward()
    grad = embedding.weight.grad
    print(f"Embedding.grad 是否存在 : {grad is not None}")
    print(f"Embedding.grad 形状     : {tuple(grad.shape)}")
    print(f"梯度的整体大小(L2)      : {grad.norm().item():.6f}")
    print("梯度告诉优化器：每个参数往哪边改、改多少。")
    print("这一步没有产生新的回答文本，权重仍未更新。")

    print()
    print("=" * 60)
    print("【3】参数更新（真正改权重的是这一步）")
    print("=" * 60)
    # SGD：weight ← weight - lr × grad。比喻：按阅卷意见改作业。
    # 不会重新算梯度；默认也不清 .grad。
    optimizer.step()
    after = embedding.weight[token_id, 0].item()
    print(f"某个权重更新前 = {before:.8f}")
    print(f"某个权重更新后 = {after:.8f}")
    print(f"差值           = {after - before:.8f}")
    print("公式直觉：weight ← weight - lr * grad")

    print()
    print("和小项目正式训练的关系：")
    print("  - scripts/sft001.py 里的 trainer.train() 才是启动多步训练循环")
    print("  - 循环里每一步仍是：Forward → Loss → Backward → optimizer.step()")
    print("  - 差别只是：数据是对话、可训练参数可能是 LoRA、优化器一般是 AdamW")

    print()
    print("小结论：")
    print("  - model.train() 是换挡（训练模式），不是踩油门。")
    print("  - 油门是 Forward + Loss + Backward + optimizer.step()。")
    print("  - Loss 衡量「预测和标签差多远」。")
    print("  - loss.backward() 只写 .grad（阅卷）；optimizer.step() 才改权重（改作业）。")
    print("  - 只有其中一步、或顺序反了、或不 zero_grad，训练会不动或乱跳。")


if __name__ == "__main__":
    main()
