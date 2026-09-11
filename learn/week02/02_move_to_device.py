"""Day 2 开头：模型和输入必须在同一块设备上。

学什么
    model.to(device) 只搬家，不训练。
    tokenizer 产出的 input_ids 默认在 CPU，要再 .to(device) 才能和模型一起算。
    设备和张量不一致时，常见报错：Expected all tensors to be on the same device。

对应阅读
    docs/学习路线/学习计划.md 第 2 周 Day 2～3
    对照 learn/week01/07_forward_logits.py（那里固定 CPU）

运行
    python learn/week02/02_move_to_device.py
"""

import torch

from _common import (
    append_run,
    find_model_dir,
    load_causal_lm,
    make_batch,
    pick_device,
    rss_mb,
)


def main():
    device = pick_device()
    model_dir = find_model_dir("135m")
    print(f"模型目录 : {model_dir}")
    print(f"目标设备 : {device}")
    print(f"加载前 RSS ≈ {rss_mb():.1f} MB")

    tokenizer, model = load_causal_lm(model_dir, device)
    param_dev = next(model.parameters()).device
    print(f"加载后 RSS ≈ {rss_mb():.1f} MB")
    print(f"权重所在 device = {param_dev}")

    print()
    print("=" * 60)
    print("【1】Tokenizer 的输出还在 CPU")
    print("=" * 60)
    cpu_batch = tokenizer("Java interface", return_tensors="pt")
    print(f"  input_ids.device（未搬） = {cpu_batch['input_ids'].device}")
    print("  这是正常的：分词在 CPU 上做，数字张量默认也在 CPU。")

    print()
    print("=" * 60)
    print("【2】送进模型前要搬到和权重同一块设备")
    print("=" * 60)
    batch = {k: v.to(device) for k, v in cpu_batch.items()}
    print(f"  input_ids.device（已搬） = {batch['input_ids'].device}")

    model.eval()
    with torch.no_grad():
        logits = model(**batch).logits
    print(f"  logits.device            = {logits.device}")
    print(f"  logits.shape             = {tuple(logits.shape)}")

    print()
    print("=" * 60)
    print("【3】make_batch 已经帮你搬好了（后面脚本都用它）")
    print("=" * 60)
    demo = make_batch(tokenizer, ["什么是多态？"], device, max_length=32)
    print(f"  labels 与 input_ids 同设备: {demo['labels'].device == demo['input_ids'].device}")
    print(f"  形状 input_ids = {tuple(demo['input_ids'].shape)}  # [batch, seq]")

    print()
    print("小结论：")
    print("  - .to(device) 是搬家，不是开始训练。")
    print("  - 模型在哪，input_ids / attention_mask / labels 就要在哪。")
    print("  - 第 1 周写死 CPU；第 2 周用 pick_device() 自动选。")

    append_run(
        {
            "script": "02_move_to_device",
            "device": str(device),
            "model_dir": str(model_dir),
            "param_device": str(param_dev),
            "rss_mb": rss_mb(),
        }
    )


if __name__ == "__main__":
    main()
