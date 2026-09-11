"""Day 2：对照 adapter（小补丁）和 merge 后的完整模型。

学什么
    训练中途/结束可以只存 LoRA 适配器：体积小，必须配原底座才能用。
    sft001 还会 merge_and_unload：把补丁加回 W，导出一份普通完整模型，
    评测脚本就能像加载 Instruct 一样 from_pretrained，不必懂 PEFT。

对应阅读
    docs/学习路线/材料/模型文件说明/01-目录与各文件总览.md 第 5 节
    scripts/sft001.py 末尾：save_pretrained(adapter) 与 merge_and_unload

运行
    python learn/week03/03_adapter_files.py

没有跑过 sft001、仓库里还没有 adapter 时，本脚本照样把概念讲完（不算不及格）。

----------------------------------------------------------------------
科普：两份产物各适合干什么

    adapter_config.json + adapter_model.safetensors
        只有补丁。继续训练、换底座版本、占磁盘少，都用它。
        加载：先底座，再 PeftModel.from_pretrained(base, adapter_dir)

    models/smollm2-135m-java/model.safetensors
        已经 merge。体积 ≈ 原 135M。评测、演示、给别人当普通模型用。
        加载：AutoModelForCausalLM.from_pretrained，和 week01 一样。

    记一句：adapter 是补丁包；merge 是把补丁缝回衣服。
"""

from __future__ import annotations

import json

from _common import (
    CHECKPOINT_DIR,
    MERGED_DIR,
    ROOT,
    append_run,
    find_adapter_dirs,
    human_mb,
    size_label,
)


def size_line(path, label: str) -> None:
    """打印一行「有/无 + 路径 + 体积」，给 adapter / merge 文件对照用。"""
    exists = path.exists()
    extra = size_label(path)
    flag = "有" if exists else "无"
    print(f"  [{flag}] {label}")
    print(f"       {path}")
    print(f"       {extra}")


def main():
    """入口：对照 adapter（小补丁）和 merge 后完整模型的位置与体积。

    没有 checkpoint 也能把概念讲完。会搜索仓库里所有 adapter_config.json，
    并把是否找到、体积写入 jsonl。
    """
    print("=" * 60)
    print("【1】本仓库约定的两个位置")
    print("=" * 60)
    adapter_final = CHECKPOINT_DIR / "adapter_final"
    print("  适配器（sft001 会写到这里）：")
    print(f"    {adapter_final}")
    print("  merge 后的完整模型（评测常用）：")
    print(f"    {MERGED_DIR}")

    print()
    print("=" * 60)
    print("【2】体积对照（有文件才显示数字）")
    print("=" * 60)
    adapter_weight = adapter_final / "adapter_model.safetensors"
    adapter_cfg = adapter_final / "adapter_config.json"
    merged_weight = MERGED_DIR / "model.safetensors"
    size_line(adapter_cfg, "adapter_config.json（r / alpha / 打在哪些层）")
    size_line(adapter_weight, "adapter_model.safetensors（只有补丁，应该很小）")
    size_line(merged_weight, "merge 后 model.safetensors（完整权重，大约几百 MB）")

    print()
    print("=" * 60)
    print("【3】全库搜索 adapter_config.json")
    print("=" * 60)
    found = find_adapter_dirs()
    if not found:
        print("  还没有适配器文件。可能还没跑 scripts/sft001.py，这很正常。")
        print("  概念仍然成立：补丁小、merge 后大。")
    else:
        for d in found:
            cfg_path = d / "adapter_config.json"
            w1 = d / "adapter_model.safetensors"
            w2 = d / "adapter_model.bin"
            w = w1 if w1.exists() else w2
            print(f"  目录: {d.relative_to(ROOT)}")
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except OSError:
                cfg = {}
            print(f"    r={cfg.get('r')}  lora_alpha={cfg.get('lora_alpha')}  "
                  f"dropout={cfg.get('lora_dropout')}")
            mods = cfg.get("target_modules")
            if mods:
                print(f"    target_modules={mods}")
            if w.exists():
                print(f"    权重约 {human_mb(w):.1f} MB  ← {w.name}")

    print()
    print("=" * 60)
    print("【4】怎么读这次对照")
    print("=" * 60)
    if adapter_weight.exists() and merged_weight.exists():
        aw = human_mb(adapter_weight) or 0
        mw = human_mb(merged_weight) or 0
        ratio = (mw / aw) if aw else None
        print(f"  补丁 {aw:.1f} MB  vs  完整 {mw:.1f} MB")
        if ratio:
            print(f"  完整大约是补丁的 {ratio:.0f} 倍。")
        print("  评测用完整模型更省事；想接着训或换底座，留着 adapter。")
    else:
        print("  缺文件时请记住这两句，Day 7 验收仍算过：")
        print("  - adapter = 只有补丁，必须配合原 Instruct 底座")
        print("  - merge 后 = 普通完整模型，eval_compare.py 直接加载")

    append_run(
        {
            "script": "03_adapter_files",
            "adapter_final_exists": adapter_final.exists(),
            "merged_exists": (MERGED_DIR / "config.json").exists(),
            "adapter_weight_mb": human_mb(adapter_weight),
            "merged_weight_mb": human_mb(merged_weight),
            "n_adapter_dirs": len(found),
        }
    )
    print()
    print("笔记作业：")
    print("  adapter 和 merge 后完整模型，各适合什么场景？各写一句话。")
    print()
    print("下一步：python learn/week03/04_compare_r.py")


if __name__ == "__main__":
    main()
