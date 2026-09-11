"""Day 4～5：同一设备上把 batch size 从 1 调到 2、4。

学什么
    batch = 一次前向同时塞进去几条样本。
    通常：batch 越大，吞吐（样本/秒）更高，但激活值占的内存也更大。
    内存不够就立刻降回去；OOM 是合法实验结果，不是失败。

对应阅读
    docs/学习路线/学习计划.md 第 2 周 Day 4～5

运行
    python learn/week02/04_batch_size.py

----------------------------------------------------------------------
科普：batch 变大，内存涨在哪

    权重（模型参数）几乎不随 batch 变：135M 就那一份。
    变大的是「中间激活」：每条样本、每个 token、每一层都有一份向量。
    所以：
        显存/内存 ≈ 权重 + batch × 序列长度 × 层数 × 隐藏维度 × 常数
    序列很长时，batch=2 也可能比短序列 batch=8 更吃内存。
    本脚本锁住 max_length=64，只改 batch，方便看趋势。
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


def main():
    device = pick_device()
    model_dir = find_model_dir("135m")
    steps = 3
    max_length = 64
    sizes = [1, 2, 4]

    print(f"设备 : {device}   模型 : {model_dir.name}")
    print(f"固定 : steps={steps}, max_length={max_length}；只改 batch size = {sizes}")
    print("内存不够时会抓住 OOM，继续试下一个更小的……本脚本是从小试到大。")

    tokenizer, model = load_causal_lm(model_dir, device)
    print()
    print(f"{'batch':<8}{'秒/步':<12}{'步/秒':<10}{'RSS MB':<10}{'设备MB':<10}{'OOM'}")

    rows = []
    for bs in sizes:
        texts = load_texts(bs)
        batch = make_batch(tokenizer, texts, device, max_length=max_length)
        stats = timed_train_steps(model, batch, steps=steps)
        stats.update(
            {
                "script": "04_batch_size",
                "device": str(device),
                "batch_size": bs,
                "max_length": max_length,
                "model": model_dir.name,
                "seq": int(batch["input_ids"].shape[1]),
            }
        )
        append_run(stats)
        rows.append(stats)
        print(
            f"{bs:<8}{fmt(stats['avg_step_s']):<12}{fmt(stats['steps_per_s']):<10}"
            f"{fmt(stats['rss_mb'], 1):<10}{fmt(stats['device_alloc_mb'], 1):<10}"
            f"{fmt(stats['oom'])}"
        )
        if stats["oom"]:
            print(f"  batch={bs} OOM，后面更大的 batch 不必再试。记下「本机稳的上限」。")
            break

    free_model(model)

    print()
    print("请把上面表格抄进笔记，并补一句：")
    print("  我这台机器上，135M + 当前设备，最稳的 batch size 是 ____")
    print()
    print("小结论：")
    print("  - batch 变大，权重几乎不变，激活变多。")
    print("  - 不够就降；不要靠「再试一次说不定行」。")
    print("  - 若还想要「更大有效 batch」又不涨那么多峰值，看下一个脚本：梯度累积。")


if __name__ == "__main__":
    main()
