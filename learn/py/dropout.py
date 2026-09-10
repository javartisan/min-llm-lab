"""Dropout 小示例：训练随机蒙眼，推理不蒙。

对照
    learn/week01/09_loss_backward.py 里的 model.train() / 07 里的 model.eval()

运行
    python learn/py/dropout.py

把 Dropout 想成：训练时临时蒙住一部分神经元，逼模型别死记某一条路。

本例 5 个数、丢掉一半（p=0.5）。注意：
    打印出来的 2.0、3.0 是向量里的数值；
    「蒙住了 2 和 3」若出现，指的是某些下标被置 0，
    不是句子的第 2、3 个字，也不是必须永远蒙住同一处。
"""

import torch


def main():
    torch.manual_seed(0)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    drop = torch.nn.Dropout(p=0.5)

    print("输入        ", x.tolist())
    print("下标         [0,    1,    2,    3,    4]")
    print()

    print("=" * 60)
    print("train()：每次随机蒙一些位置，留下的会放大")
    print("=" * 60)
    drop.train()
    for i in range(1, 4):
        y = drop(x)
        print(f"第{i}次", y.tolist())
    print()
    print("p=0.5 时，留下的乘 1/(1-0.5)=2，所以 1→2、4→8、5→10。")
    print("这叫 inverted dropout，让「训练时的平均值」接近「推理全开」。")
    print("每次蒙的下标都可能不同，不是固定蒙住某两个数。")

    print()
    print("=" * 60)
    print("eval()：什么都不蒙，结果稳定")
    print("=" * 60)
    drop.eval()
    print("第1次", drop(x).tolist())
    print("第2次", drop(x).tolist())
    print("推理不会沿用某一次训练的那组蒙眼位置，而是用全部神经元。")

    print()
    print("小结论：")
    print("  - 训练随机残缺，推理走完整路，这是设计，不是漏蒙。")
    print("  - 若推理还 train()，每次蒙的位置不同，预测会乱跳。")
    print("  - SmolLM2 的 dropout 概率是 0，数值上几乎看不出差别。")


if __name__ == "__main__":
    main()
