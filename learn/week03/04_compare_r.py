"""Day 3～5：只改 r（8 vs 16），其它与 sft001 对齐。

学什么
    r 是 LoRA 的秩：旁路矩阵有多「厚」。
    r 越大 → A/B 参数越多 → 更能拟合任务，也更占内存、小数据更容易过拟合。

对应阅读
    docs/学习路线/学习计划.md 第 3 周 Day 3～5
    scripts/sft001.py：r=16

运行
    python learn/week03/04_compare_r.py

----------------------------------------------------------------------
科普：一次只拧一颗螺丝

    正式调参最忌讳同时改 r、dropout、target_modules。
    本周三个脚本各改一个变量，对照才读得懂。

    本脚本固定：
        lora_alpha=32, lora_dropout=0.05, target_modules=注意力+MLP

    只改 r=8 与 r=16。

    副作用（要心里有数）：
        缩放 = alpha / r
        r=16 → 2
        r=8  → 4
        所以这次对比里，「参数量」和「旁路放大」其实一起变了。
        有人会把 alpha 设成 2*r 来保持缩放不变；我们跟 sft001，不额外改 alpha。

    极少步的 last_loss 不能当「谁更好」的证据，只看：
        可训练参数是否按预期涨、能否跑通、有没有 OOM。
"""

from _common import ALL_MODULES, fmt, run_lora_tiny


def main():
    print("固定：alpha=32, dropout=0.05, 模块=注意力+MLP")
    print(f"模块列表：{ALL_MODULES}")
    print("只改 r。每次重新加载 135M，避免两次实验互相污染。")
    print()

    rows = []
    for r in (8, 16):
        print(f"—— r={r}  缩放={32 / r:.1f} ——")
        rec = run_lora_tiny(
            script="04_compare_r",
            tag=f"r={r}",
            r=r,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=ALL_MODULES,
        )
        rows.append(rec)

    print()
    print(f"{'r':<8}{'可训练':<16}{'占比%':<10}{'缩放':<8}{'秒/步':<10}{'loss':<10}{'OOM'}")
    for rec in rows:
        print(
            f"{rec['r']:<8}{rec['lora_trainable']:<16,}"
            f"{rec['trainable_pct']:<10}"
            f"{rec['scale_alpha_over_r']:<8.1f}"
            f"{fmt(rec['avg_step_s']):<10}"
            f"{fmt(rec['last_loss']):<10}"
            f"{fmt(rec['oom'])}"
        )

    if len(rows) == 2 and rows[0]["lora_trainable"] and rows[1]["lora_trainable"]:
        a, b = rows[0]["lora_trainable"], rows[1]["lora_trainable"]
        print()
        print(f"可训练参数：r=16 大约是 r=8 的 {b / a:.2f} 倍（秩加倍，A/B 也大约加倍）。")

    print()
    print("请抄进笔记：")
    print("  r 从 8 → 16，可训练参数从 ____ 到 ____")
    print("  我更愿意在小数据上先用哪个 r？为什么？")
    print()
    print("下一步：python learn/week03/05_compare_dropout.py")


if __name__ == "__main__":
    main()
