"""Day 2～3：同一 135M、同一数据、同一 batch=1，对比 CPU 与 MPS。

学什么
    公平对比要锁住：模型、数据、步数、batch size，只改设备。
    记录：耗时（或步/秒）、内存、是否 OOM。
    MPS 不可用时，CPU 那一行照样填，MPS 行写「不可用」。

对应阅读
    docs/学习路线/学习计划.md 第 2 周 Day 2～3 的对比表

运行
    python learn/week02/03_cpu_vs_mps.py

----------------------------------------------------------------------
科普：为什么必须「同一模型同一数据」

    如果 CPU 用 135M、MPS 用 360M，快慢差的是模型大小，不是设备。
    如果 batch 不同，吃内存的是 batch，也会搅乱对比。
    本脚本固定：
        模型 = 本地 135M Instruct（或回退的同系列）
        文本 = train.jsonl 前几条
        batch size = 1
        训练步数 = 3（外加 1 步热身，不计入平均）
"""

import torch

from _common import (
    append_run,
    find_model_dir,
    fmt,
    free_model,
    load_causal_lm,
    load_texts,
    make_batch,
    mps_available,
    rss_mb,
    timed_train_steps,
)


def run_on(device_name: str, model_dir, texts, steps: int, max_length: int) -> dict:
    device = torch.device(device_name)
    print()
    print("=" * 60)
    print(f"设备 {device_name}  加载中…")
    print("=" * 60)
    tokenizer, model = load_causal_lm(model_dir, device)
    batch = make_batch(tokenizer, texts, device, max_length=max_length)
    print(f"  batch shape = {tuple(batch['input_ids'].shape)}  # [bs, seq]")
    stats = timed_train_steps(model, batch, steps=steps)
    stats.update(
        {
            "script": "03_cpu_vs_mps",
            "device": device_name,
            "batch_size": 1,
            "max_length": max_length,
            "model": str(model_dir.name),
        }
    )
    print(f"  是否 OOM        = {fmt(stats['oom'])}")
    print(f"  平均每步耗时    = {fmt(stats['avg_step_s'])} s")
    print(f"  大约步/秒       = {fmt(stats['steps_per_s'])}")
    print(f"  进程 RSS        = {fmt(stats['rss_mb'], 1)} MB")
    print(f"  设备已分配内存  = {fmt(stats['device_alloc_mb'], 1)} MB")
    print(f"  最后一步 loss   = {fmt(stats['last_loss'])}")
    free_model(model)
    append_run(stats)
    return stats


def main():
    steps = 3
    max_length = 64
    model_dir = find_model_dir("135m")
    texts = load_texts(1)  # batch size = 1
    print(f"模型 : {model_dir}")
    print(f"固定 : batch=1, steps={steps}, max_length={max_length}")
    print(f"样本 : {texts}")
    print(f"开始前 RSS ≈ {rss_mb():.1f} MB")

    rows = []
    rows.append(run_on("cpu", model_dir, texts, steps, max_length))

    if mps_available():
        rows.append(run_on("mps", model_dir, texts, steps, max_length))
    else:
        print()
        print("=" * 60)
        print("MPS 不可用，本行不跑")
        print("=" * 60)
        print("  把计划表 MPS 一行填「不可用」即可。概念用 04 的 CPU 不同 batch 继续学。")
        append_run(
            {
                "script": "03_cpu_vs_mps",
                "device": "mps",
                "batch_size": 1,
                "skipped": True,
                "reason": "mps_not_available",
            }
        )

    print()
    print("=" * 60)
    print("对照表（抄到笔记里）")
    print("=" * 60)
    print(f"{'设备':<8}{'batch':<8}{'秒/步':<12}{'步/秒':<10}{'RSS MB':<10}{'OOM'}")
    for r in rows:
        print(
            f"{r['device']:<8}{r['batch_size']:<8}"
            f"{fmt(r['avg_step_s']):<12}{fmt(r['steps_per_s']):<10}"
            f"{fmt(r['rss_mb'], 1):<10}{fmt(r['oom'])}"
        )

    print()
    print("小结论：")
    print("  - 只改设备，才能说「这块硬件更快/更省」。")
    print("  - RSS 是进程高水位，比较粗糙；OOM 比精确数字更重要。")
    print("  - 这几步不是认真微调，loss 大小不必和 sft001 比。")


if __name__ == "__main__":
    main()
