"""Day 4～5：梯度累积 —— 用多次小 batch 模拟一次大 batch。

学什么
    有效 batch ≈ 每次前向的 batch_size × 累积步数 accum。
    做法：连续几次 backward（梯度加在一起），再 optimizer.step() 一次。
    峰值内存接近「小 batch」，优化器更新频率接近「大 batch」。

对应阅读
    docs/学习路线/学习计划.md 第 2 周 Day 4～5（gradient accumulation）
    对照 learn/week01/09_loss_backward.py：backward 会把梯度累加到 .grad

运行
    python learn/week02/05_grad_accum.py

----------------------------------------------------------------------
科普：为什么 backward 不 zero_grad 就会累加

    第 1 周已经见过：.grad 默认是累加的，所以每步前要 zero_grad。
    梯度累积是故意延后 zero_grad / step：
        micro 1: loss/accum → backward     # .grad 有了第一份
        micro 2: loss/accum → backward     # 再加第二份
        optimizer.step()                   # 按总和更新一次
        optimizer.zero_grad()

    除以 accum 是为了让「两次小步加起来」的梯度尺度，
    近似「一次把两条样本拼成 batch=2」的平均梯度。

    和真·batch=2 不完全相同（BatchNorm 等会有差别；本模型几乎不用 BN），
    但对 Causal LM + 本仓库这种小实验，直觉够用。
"""

from _common import (
    append_run,
    find_model_dir,
    fmt,
    free_model,
    load_causal_lm,
    load_texts,
    make_batch,
    pick_device,
    timed_train_steps,
)


def run_cfg(model, tokenizer, device, *, batch_size, accum, steps, max_length):
    texts = load_texts(batch_size)
    batch = make_batch(tokenizer, texts, device, max_length=max_length)
    stats = timed_train_steps(model, batch, steps=steps, accum=accum)
    effective = batch_size * accum
    stats.update(
        {
            "script": "05_grad_accum",
            "device": str(device),
            "batch_size": batch_size,
            "accum": accum,
            "effective_batch": effective,
            "max_length": max_length,
        }
    )
    append_run(stats)
    print(
        f"  bs={batch_size}  accum={accum}  有效batch={effective:<4}"
        f"秒/步={fmt(stats['avg_step_s']):<8} RSS={fmt(stats['rss_mb'], 1):<8}"
        f"OOM={fmt(stats['oom'])}"
    )
    return stats


def main():
    device = pick_device()
    model_dir = find_model_dir("135m")
    steps = 3
    max_length = 64
    print(f"设备 : {device}   模型 : {model_dir.name}")
    print("对比三档（有效 batch 都想接近 4）：")
    print("  A) batch=4, accum=1  → 一次塞 4 条，内存最猛")
    print("  B) batch=2, accum=2  → 两次小步再更新")
    print("  C) batch=1, accum=4  → 四次更小的步再更新")
    print()

    tokenizer, model = load_causal_lm(model_dir, device)
    print(f"{'配置':<24}结果")
    run_cfg(model, tokenizer, device, batch_size=4, accum=1, steps=steps, max_length=max_length)
    run_cfg(model, tokenizer, device, batch_size=2, accum=2, steps=steps, max_length=max_length)
    run_cfg(model, tokenizer, device, batch_size=1, accum=4, steps=steps, max_length=max_length)
    free_model(model)

    print()
    print("怎么读：")
    print("  - 若 A OOM、C 能跑：这就是累积的意义——用时间换峰值内存。")
    print("  - 秒/步：C 往往更慢（同一次 step 里做了 4 次前向）。")
    print("  - 正式训练里 SFTConfig 的 gradient_accumulation_steps 就是这件事。")
    print()
    print("小结论：")
    print("  - 有效 batch = micro_batch × accum。")
    print("  - 内存看 micro_batch；更新频率看有效 batch。")


if __name__ == "__main__":
    main()
