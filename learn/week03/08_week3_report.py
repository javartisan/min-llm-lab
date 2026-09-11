"""Day 7：把本周 LoRA 实验收成一张表 + 短笔记。

学什么
    验收三件事：能画冻底座+旁路、能区分 adapter/merge、至少一组超参对比。
    数字来自你自己的机器；结论用自己的话写进 reports/week03_lora.md。

对应阅读
    docs/学习路线/学习计划.md 第 3 周 Day 7 与验收清单

运行
    python learn/week03/08_week3_report.py

会读取 01～07 追加的 reports/week03_runs.jsonl，
并写出 reports/week03_lora.md 方便以后回看。
"""

from __future__ import annotations

import json

from _common import REPORTS_DIR, RUNS_FILE, fmt


def load_runs() -> list[dict]:
    """读 reports/week03_runs.jsonl：每行一条实验记录。文件不存在则空列表。"""
    if not RUNS_FILE.exists():
        return []
    rows = []
    with RUNS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_script(rows: list[dict], script: str) -> list[dict]:
    """筛出某个脚本名（如 04_compare_r）写下的全部记录，保持文件中的顺序。"""
    return [r for r in rows if r.get("script") == script]


def main():
    """入口：汇总本周 jsonl，打印对照表和验收口答，并写出 reports/week03_lora.md。

    还没跑实验时会提示先跑 01～06。三句话总结要自己动手填。
    """
    rows = load_runs()
    print("=" * 60)
    print("第 3 周复盘")
    print("=" * 60)
    print(f"记录文件: {RUNS_FILE}")
    print(f"已有记录 {len(rows)} 条")
    if not rows:
        print()
        print("还没有实验记录。请先按编号跑 01～06，再回到本脚本。")
        print("  python learn/week03/01_lora_idea.py")
        print("  python learn/week03/02_inject_and_count.py")
        return

    print()
    print("-" * 60)
    print("Day 1  注入后可训练参数")
    print("-" * 60)
    for r in by_script(rows, "02_inject_and_count"):
        print(
            f"  模型={r.get('model')}  全量可训练={r.get('full_trainable')}  "
            f"LoRA 可训练={r.get('lora_trainable')}  占比%={r.get('trainable_pct')}"
        )
    if not by_script(rows, "02_inject_and_count"):
        print("  尚未运行 02。")

    print()
    print("-" * 60)
    print("Day 2  adapter vs merge")
    print("-" * 60)
    rec3 = by_script(rows, "03_adapter_files")
    if not rec3:
        print("  尚未运行 03。没有真实文件也可以，概念仍要能口述。")
    for r in rec3:
        print(
            f"  adapter_final 存在={r.get('adapter_final_exists')}  "
            f"merge 模型存在={r.get('merged_exists')}"
        )
        print(
            f"  补丁约 {fmt(r.get('adapter_weight_mb'), 1)} MB  "
            f"完整约 {fmt(r.get('merged_weight_mb'), 1)} MB"
        )

    print()
    print("-" * 60)
    print("Day 3～5  超参对照（每次只改一个）")
    print("-" * 60)
    print(f"{'脚本':<22}{'标签':<16}{'可训练':<14}{'秒/步':<10}{'OOM'}")
    for script in ("04_compare_r", "05_compare_dropout", "06_compare_targets"):
        recs = by_script(rows, script)
        if not recs:
            print(f"{script:<22}{'（未跑）':<16}")
        for r in recs:
            print(
                f"{script:<22}{str(r.get('tag', '')):<16}"
                f"{str(r.get('lora_trainable', '')):<14}"
                f"{fmt(r.get('avg_step_s')):<10}"
                f"{fmt(r.get('oom'))}"
            )

    print()
    print("-" * 60)
    print("Day 6  1.7B（可选）")
    print("-" * 60)
    rec7 = by_script(rows, "07_try_1p7b")
    if not rec7:
        print("  未运行 07。可选，失败也不扣分。")
    for r in rec7:
        print(
            f"  {r.get('model')}  load_ok={r.get('load_ok')}  skipped={r.get('skipped', False)}"
            f"  秒/步={fmt(r.get('avg_step_s'))}  OOM={fmt(r.get('oom'))}"
        )
        if r.get("error"):
            print(f"    原因: {r['error'][:180]}")

    print()
    print("-" * 60)
    print("对照  135M 全量 SFT（09）")
    print("-" * 60)
    rec9 = by_script(rows, "09_full_sft_135m")
    if not rec9:
        print("  未运行 09。可选：python learn/week03/09_full_sft_135m.py")
    for r in rec9:
        print(
            f"  {r.get('model')}  可训练={r.get('full_trainable')}  "
            f"占比%={r.get('trainable_pct')}  loss={fmt(r.get('train_loss'))}"
        )
        print(f"  输出: {r.get('output')}")

    print()
    print("=" * 60)
    print("验收口答（先自己答，再对答案方向）")
    print("=" * 60)
    print("Q: 为什么 LoRA 能省参数？")
    print("A: 冻住大矩阵 W，只训低秩 A、B；参数从 out×in 变成 r×(in+out)。")
    print()
    print("Q: 为什么小数据更适合 LoRA？")
    print("A: 能改的旋钮少，不容易把那几条样本背下来；底座通用能力尽量留下。")
    print()
    print("Q: merge 之后评测为何更方便？")
    print("A: 变成普通完整模型，不必再配 PEFT/底座双路径，eval 脚本和 Instruct 一样加载。")
    print()
    print("Q: adapter 和 merge 差在哪？")
    print("A: adapter 只有补丁、体积小、要配底座；merge 已缝回去、体积≈原模型。")

    md_path = REPORTS_DIR / "week03_lora.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 第 3 周：LoRA 结论",
        "",
        "冻底座 + 旁路示意（自己补画）：",
        "",
        "```",
        "x → [冻住的 W] → + → y",
        "     ↘ [A]→[B] ↗",
        "```",
        "",
        "## 注入后参数",
        "",
        "| 模型 | 全量可训练 | LoRA 可训练 | 占比% |",
        "|---|---|---|---|",
    ]
    for r in by_script(rows, "02_inject_and_count"):
        lines.append(
            f"| {r.get('model')} | {r.get('full_trainable')} | "
            f"{r.get('lora_trainable')} | {r.get('trainable_pct')} |"
        )
    lines += [
        "",
        "## adapter vs merge",
        "",
        f"- adapter 权重 MB: 见 03 记录",
        f"- merge 权重 MB: 见 03 记录",
        "",
        "## 超参对照",
        "",
        "| 脚本 | 标签 | r | dropout | 模块数 | 可训练 | 秒/步 | OOM |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for script in ("04_compare_r", "05_compare_dropout", "06_compare_targets"):
        for r in by_script(rows, script):
            nmod = r.get("n_target_modules", "")
            lines.append(
                f"| {script} | {r.get('tag')} | {r.get('r')} | {r.get('lora_dropout')} "
                f"| {nmod} | {r.get('lora_trainable')} | {fmt(r.get('avg_step_s'))} "
                f"| {fmt(r.get('oom'))} |"
            )
    lines += [
        "",
        "## 1.7B（可选）",
        "",
    ]
    if rec7:
        for r in rec7:
            lines.append(
                f"- {r.get('model')} load_ok={r.get('load_ok')} "
                f"OOM={fmt(r.get('oom'))} error={r.get('error', '')[:120]}"
            )
    else:
        lines.append("- 未跑。")
    lines += [
        "",
        "## 135M 全量 SFT（09，对照 LoRA）",
        "",
    ]
    if rec9:
        for r in rec9:
            lines.append(
                f"- {r.get('model')} 可训练={r.get('full_trainable')} "
                f"loss={fmt(r.get('train_loss'))} 输出={r.get('output')}"
            )
    else:
        lines.append("- 未跑。可选：`python learn/week03/09_full_sft_135m.py`")
    lines += [
        "",
        "## 三句话总结（自己填）",
        "",
        "1. ",
        "2. ",
        "3. ",
        "",
        "## 验收自勾",
        "",
        "- [ ] 能画冻底座 + LoRA 旁路",
        "- [ ] 能解释 adapter vs merge",
        "- [ ] 至少完成 1 组超参对比（04 / 05 / 06 任一）",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"已写出 {md_path}")
    print("请打开文件，把示意、三句话和验收勾选补全。")


if __name__ == "__main__":
    main()
