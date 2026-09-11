"""Day 7：把本周实验收成一张表 + 短笔记。

学什么
    验收三件事：会测设备、能解释 batch 与内存、有自己的速度/内存表。
    「我的机器上最稳的配置」比追最高吞吐更重要。

对应阅读
    docs/学习路线/学习计划.md 第 2 周 Day 7 与验收清单

运行
    python learn/week02/07_week2_report.py

会读取 01～06 追加的 reports/week02_runs.jsonl，
并写出 reports/week02_device.md 方便以后回看。
"""

from __future__ import annotations

import json

from _common import RUNS_FILE, device_report, fmt, mps_available, pick_device, REPORTS_DIR


def load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    rows = []
    with RUNS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_by_script(rows: list[dict], script: str) -> list[dict]:
    return [r for r in rows if r.get("script") == script]


def main():
    info = device_report()
    rows = load_runs()
    print("=" * 60)
    print("第 2 周复盘")
    print("=" * 60)
    print(f"torch={info['torch']}  MPS available={info['mps_is_available']}  默认设备={info['picked']}")
    print(f"记录文件: {RUNS_FILE}")
    print(f"已有记录 {len(rows)} 条")
    if not rows:
        print()
        print("还没有实验记录。请先按编号跑 01～06，再回到本脚本。")
        print("  python learn/week02/01_probe_device.py")
        print("  python learn/week02/03_cpu_vs_mps.py")
        print("  python learn/week02/04_batch_size.py")
        return

    print()
    print("-" * 60)
    print("Day 2～3  CPU vs MPS（batch=1）")
    print("-" * 60)
    print(f"{'设备':<8}{'秒/步':<12}{'RSS MB':<10}{'OOM':<8}备注")
    for r in latest_by_script(rows, "03_cpu_vs_mps"):
        note = r.get("reason") or r.get("model", "")
        if r.get("skipped"):
            print(f"{r.get('device', ''):<8}{'—':<12}{'—':<10}{'—':<8}{note}")
        else:
            print(
                f"{str(r.get('device', '')):<8}{fmt(r.get('avg_step_s')):<12}"
                f"{fmt(r.get('rss_mb'), 1):<10}{fmt(r.get('oom')):<8}{note}"
            )

    print()
    print("-" * 60)
    print("Day 4～5  batch size")
    print("-" * 60)
    print(f"{'设备':<8}{'batch':<8}{'秒/步':<12}{'RSS MB':<10}{'OOM'}")
    for r in latest_by_script(rows, "04_batch_size"):
        print(
            f"{str(r.get('device', '')):<8}{str(r.get('batch_size', '')):<8}"
            f"{fmt(r.get('avg_step_s')):<12}{fmt(r.get('rss_mb'), 1):<10}{fmt(r.get('oom'))}"
        )

    print()
    print("-" * 60)
    print("梯度累积")
    print("-" * 60)
    print(f"{'bs':<6}{'accum':<8}{'有效bs':<10}{'秒/步':<12}{'RSS':<10}{'OOM'}")
    for r in latest_by_script(rows, "05_grad_accum"):
        print(
            f"{str(r.get('batch_size', '')):<6}{str(r.get('accum', '')):<8}"
            f"{str(r.get('effective_batch', '')):<10}{fmt(r.get('avg_step_s')):<12}"
            f"{fmt(r.get('rss_mb'), 1):<10}{fmt(r.get('oom'))}"
        )

    print()
    print("-" * 60)
    print("360M（可选）")
    print("-" * 60)
    rec360 = latest_by_script(rows, "06_try_360m")
    if not rec360:
        print("  尚未运行 06。可选：python learn/week02/06_try_360m.py")
    for r in rec360:
        print(
            f"  {r.get('model')}  device={r.get('device')}  load_ok={r.get('load_ok', True)}"
            f"  秒/步={fmt(r.get('avg_step_s'))}  OOM={fmt(r.get('oom'))}"
        )
        if r.get("error"):
            print(f"    原因: {r['error'][:180]}")

    print()
    print("=" * 60)
    print("验收口答")
    print("=" * 60)
    print("Q: 怎么测设备可用性？")
    print("A: torch.__version__、mps.is_available()、pick_device()。见 01。")
    print()
    print("Q: batch 和内存什么关系？")
    print("A: 权重大致固定；激活随 batch×序列长度涨。不够就降 batch，或用累积。")
    print()
    print("Q: 我这台机器最稳的配置？")
    print(f"A: 请根据上表自己填。默认设备现在是 {pick_device()}；")
    print("   MPS 不可用就写 CPU + 某个没 OOM 的 batch。")

    if not mps_available():
        print()
        print("提醒：本机 MPS 不可用，用 CPU 不同 batch 完成概念目标即可。")

    md_path = REPORTS_DIR / "week02_device.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 第 2 周：我的训练设备结论",
        "",
        f"- torch: `{info['torch']}`",
        f"- MPS available: `{info['mps_is_available']}`",
        f"- 默认设备: `{info['picked']}`",
        "",
        "## 对比摘录",
        "",
        "| 来源 | 设备 | batch | accum | 秒/步 | RSS MB | OOM |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("script") in {"03_cpu_vs_mps", "04_batch_size", "05_grad_accum", "06_try_360m"}:
            lines.append(
                f"| {r.get('script')} | {r.get('device', '')} | {r.get('batch_size', '')} "
                f"| {r.get('accum', '')} | {fmt(r.get('avg_step_s'))} | {fmt(r.get('rss_mb'), 1)} "
                f"| {fmt(r.get('oom'))} |"
            )
    lines += [
        "",
        "## 最稳配置（自己填）",
        "",
        "- 设备：",
        "- batch size：",
        "- 是否用梯度累积：",
        "- 135M / 360M：",
        "",
        "三句话总结：",
        "",
        "1. ",
        "2. ",
        "3. ",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"已写出 {md_path}")
    print("请打开文件，把「最稳配置」和三句话补全。")


if __name__ == "__main__":
    main()
