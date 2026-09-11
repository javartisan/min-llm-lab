"""Day 6：尝试更大一点的模型 SmolLM2-360M-Instruct（可选）。

----------------------------------------------------------------------
本脚本主要学习目的（一句话）

    在「设备、batch、序列长度都固定」的前提下，只把模型从 135M 换成 360M，
    亲手感受：参数量变大 → 权重更重、同样一步训练更吃内存/更慢。

学什么（拆开记）
    1. 规模直觉：135M → 360M 大约 2.7 倍参数；不是「换架构」，只是更大同系模型。
    2. 对照实验：先跑 135M 作基线，再跑 360M；对比秒/步、RSS、是否 OOM。
    3. 机器边界：能加载 + 极少步跑通就算达标；OOM / 太慢要记下原因，不算不及格。
    4. 决策习惯：本机撑不住就退回 135M，或下周用 LoRA 只训小补丁，而不是硬扛全量。

不学什么
    不追求把 360M 训好；不改 batch / 学习率做调参竞赛。
    没有本地 360M 时，可先用 135M 跑完 03～05，再决定要不要 --download。

对应阅读
    docs/学习路线/学习计划.md 第 2 周 Day 6
    docs/学习路线/材料/第2周-BaseAutoModelClass与Auto家族.md（仍是 AutoModelForCausalLM）

运行
    python learn/week02/06_try_360m.py
    python learn/week02/06_try_360m.py --download   # 本地没有时才从镜像拉

模型目录
    models/SmolLM2-360M-Instruct
    Hugging Face id: HuggingFaceTB/SmolLM2-360M-Instruct
"""

from __future__ import annotations

import argparse
import sys

from _common import (
    HF_360,
    MODELS_360,
    ROOT,
    append_run,
    find_model_dir,
    fmt,
    free_model,
    load_causal_lm,
    load_texts,
    make_batch,
    patch_hf_mirror,
    pick_device,
    timed_train_steps,
)


def try_find_360():
    try:
        return find_model_dir("360m")
    except FileNotFoundError:
        return None


def download_360m():
    patch_hf_mirror()
    dest = ROOT / "models" / "SmolLM2-360M-Instruct"
    print(f"从 {HF_360} 下载到 {dest} …")
    print("（走 HF_ENDPOINT 镜像；体积比 135M 大，需要一些时间和磁盘。）")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_360)
    model = AutoModelForCausalLM.from_pretrained(HF_360)
    dest.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(dest)
    model.save_pretrained(dest)
    del model
    print("下载完成。")
    return dest


def bench(tag: str, model_dir, device, steps: int, max_length: int) -> dict:
    print()
    print("=" * 60)
    print(f"{tag}  {model_dir.name}  → {device}")
    print("=" * 60)
    try:
        tokenizer, model = load_causal_lm(model_dir, device)
    except Exception as e:
        rec = {
            "script": "06_try_360m",
            "model": model_dir.name,
            "device": str(device),
            "load_ok": False,
            "oom": "memory" in str(e).lower() or "oom" in str(e).lower(),
            "error": str(e),
        }
        append_run(rec)
        print(f"  加载失败: {e}")
        return rec

    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量约 {nparams:.1f} M")
    batch = make_batch(tokenizer, load_texts(1), device, max_length=max_length)
    stats = timed_train_steps(model, batch, steps=steps)
    stats.update(
        {
            "script": "06_try_360m",
            "model": model_dir.name,
            "device": str(device),
            "batch_size": 1,
            "max_length": max_length,
            "load_ok": True,
            "params_m": round(nparams, 1),
        }
    )
    append_run(stats)
    print(
        f"  秒/步={fmt(stats['avg_step_s'])}  RSS={fmt(stats['rss_mb'], 1)} MB"
        f"  OOM={fmt(stats['oom'])}"
    )
    if stats.get("error"):
        print(f"  错误: {stats['error'][:200]}")
    free_model(model)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="本地没有 360M 时下载")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    device = pick_device()
    dir_360 = try_find_360()
    if dir_360 is None:
        print("本地还没有 360M。候选目录：")
        for p in MODELS_360:
            print(f"  {p}")
        if args.download:
            try:
                dir_360 = download_360m()
            except Exception as e:
                print(f"下载失败: {e}")
                append_run(
                    {
                        "script": "06_try_360m",
                        "model": "SmolLM2-360M-Instruct",
                        "load_ok": False,
                        "error": f"download: {e}",
                    }
                )
                print("不及格？不算。记下失败原因即可。135M 的对比已经覆盖本周概念。")
                sys.exit(0)
        else:
            print()
            print("先用 135M 对照，需要 360M 时再加 --download：")
            print("  python learn/week02/06_try_360m.py --download")
            print()

    dir_135 = find_model_dir("135m")
    print(f"设备默认: {device}")
    print("同一套实验：batch=1、短序列、极少步。只换模型大小。")
    bench("135M", dir_135, device, args.steps, args.max_length)

    if dir_360 is not None:
        bench("360M", dir_360, device, args.steps, args.max_length)
    else:
        print("跳过 360M 实测（未下载）。")

    print()
    print("小结论：")
    print("  - 更大模型：权重更大，同样 batch 的激活也更大。")
    print("  - 跑不通就降回 135M，或下周用 LoRA 只训补丁。")
    print("  - 失败原因写进笔记，07 复盘会用到。")


if __name__ == "__main__":
    main()
