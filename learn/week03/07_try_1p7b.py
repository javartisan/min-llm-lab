"""Day 6：可选挑战 —— 1.7B + LoRA 极少步。

本脚本主要学习目的（一句话）
    体会「模型变大时，全量微调不现实，LoRA 才是入场券」；
    能加载并跑通极小步就算达标。OOM / 太慢 / 下载失败都要记下，不算不及格。

学什么
    1.7B 权重本身就比 135M 大约一个数量级；全量更新还要再存优化器状态。
    挂上 LoRA 后，可训练参数仍是百万级补丁，底座冻住。
    本机（Intel Mac）很可能仍然吃力——这正是第 4 周 QLoRA 要解决的问题。

不学什么
    不追求把 1.7B 训好；不和 135M 拼效果。

对应阅读
    docs/学习路线/学习计划.md 第 3 周 Day 6

运行
    python learn/week03/07_try_1p7b.py
    python learn/week03/07_try_1p7b.py --download   # 本地没有时才从镜像拉（体积很大）

模型
    Hugging Face: HuggingFaceTB/SmolLM2-1.7B-Instruct
    本地目录: models/SmolLM2-1.7B-Instruct
"""

from __future__ import annotations

import argparse
import sys

from _common import (
    ALL_MODULES,
    HF_1P7,
    MODELS_1P7,
    ROOT,
    append_run,
    count_params,
    find_model_dir,
    fmt,
    free_model,
    inject_lora,
    load_causal_lm,
    load_texts,
    make_batch,
    make_lora_config,
    patch_hf_mirror,
    pct,
    pick_device,
    timed_train_steps,
)


def try_find_1p7():
    try:
        return find_model_dir("1.7b")
    except FileNotFoundError:
        return None


def download_1p7():
    patch_hf_mirror()
    dest = ROOT / "models" / "SmolLM2-1.7B-Instruct"
    print(f"从 {HF_1P7} 下载到 {dest} …")
    print("体积远大于 135M，需要时间和磁盘。可随时 Ctrl+C。")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_1P7)
    model = AutoModelForCausalLM.from_pretrained(HF_1P7)
    dest.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(dest)
    model.save_pretrained(dest)
    del model
    print("下载完成。")
    return dest


def bench(model_dir, device, steps: int, max_length: int) -> dict:
    print()
    print("=" * 60)
    print(f"加载 {model_dir.name}  → {device}")
    print("=" * 60)
    try:
        tokenizer, model = load_causal_lm(model_dir, device)
    except Exception as e:
        rec = {
            "script": "07_try_1p7b",
            "model": model_dir.name,
            "device": str(device),
            "load_ok": False,
            "oom": "memory" in str(e).lower() or "oom" in str(e).lower(),
            "error": str(e),
        }
        append_run(rec)
        print(f"  加载失败: {e}")
        print("  记下瓶颈即可。135M 上的 02～06 已经覆盖 LoRA 原理。")
        return rec

    total, train = count_params(model)
    print(f"  挂 LoRA 前：总参数 {total:,}  可训练 {train:,}")
    model = inject_lora(model, make_lora_config(target_modules=ALL_MODULES))
    total2, train2 = count_params(model)
    print(f"  挂 LoRA 后：总参数 {total2:,}  可训练 {train2:,}  ({pct(train2, total2):.3f}%)")

    batch = make_batch(tokenizer, load_texts(1), device, max_length=max_length)
    stats = timed_train_steps(model, batch, steps=steps)
    rec = {
        "script": "07_try_1p7b",
        "model": model_dir.name,
        "device": str(device),
        "load_ok": True,
        "full_params": total,
        "lora_trainable": train2,
        "trainable_pct": round(pct(train2, total2), 4),
        "batch_size": 1,
        "max_length": max_length,
        **stats,
    }
    append_run(rec)
    print(
        f"  秒/步={fmt(stats['avg_step_s'])}  RSS={fmt(stats['rss_mb'], 1)} MB"
        f"  OOM={fmt(stats['oom'])}  loss={fmt(stats['last_loss'])}"
    )
    if stats.get("error"):
        print(f"  错误: {stats['error'][:200]}")
    free_model(model)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="本地没有 1.7B 时下载")
    parser.add_argument("--steps", type=int, default=1, help="极大模型默认只跑 1 步")
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()

    device = pick_device()
    print(f"设备默认: {device}")
    print("1.7B 全量微调在本机通常不现实；这里只验证「LoRA 极少步」能不能站住。")

    dir_1p7 = try_find_1p7()
    if dir_1p7 is None:
        print("本地还没有 1.7B。候选目录：")
        for p in MODELS_1P7:
            print(f"  {p}")
        if args.download:
            try:
                dir_1p7 = download_1p7()
            except Exception as e:
                print(f"下载失败: {e}")
                append_run(
                    {
                        "script": "07_try_1p7b",
                        "model": "SmolLM2-1.7B-Instruct",
                        "load_ok": False,
                        "error": f"download: {e}",
                    }
                )
                print("不算不及格。瓶颈写进笔记：下载 / 磁盘 / 网络。")
                sys.exit(0)
        else:
            print()
            print("需要权重时再加 --download（体积很大，先确认磁盘）：")
            print("  python learn/week03/07_try_1p7b.py --download")
            print("跳过 1.7B 实测也可以直接去 08 复盘。")
            append_run(
                {
                    "script": "07_try_1p7b",
                    "model": "SmolLM2-1.7B-Instruct",
                    "load_ok": False,
                    "skipped": True,
                    "error": "local weights missing; pass --download to fetch",
                }
            )
            return

    bench(dir_1p7, device, args.steps, args.max_length)

    print()
    print("小结论：")
    print("  - 底座再大，LoRA 要训的仍是小补丁；但底座本身仍要放进内存。")
    print("  - 底座都放不下时，下一周才轮到 QLoRA（先量化再挂补丁）。")
    print("  - 失败原因留给 08 复盘，目标是认知，不是刷 1.7B 分数。")


if __name__ == "__main__":
    main()
