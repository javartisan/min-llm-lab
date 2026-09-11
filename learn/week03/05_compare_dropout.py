"""Day 3～5：只改 lora_dropout（0 vs 0.05）。

学什么
    lora_dropout 只作用在 LoRA 旁路上：训练时随机丢掉一部分旁路信号。
    目的和普通 Dropout 一样——别死记训练句。推理 / eval 时会关掉。

对应阅读
    scripts/sft001.py：lora_dropout=0.05
    learn/week01/09 与 learn/py/dropout.py（Dropout 开关）

运行
    python learn/week03/05_compare_dropout.py

----------------------------------------------------------------------
科普：3 步看不出过拟合，但能看清「改了什么」

    Dropout 不增加 A/B 的参数个数，所以两边可训练参数应该几乎一样。
    差在训练时旁路会不会随机变瘦。

    本仓库 Java 数据很少，0.05 是一个温和的防过拟合旋钮，不是魔法。
    若数据再少、r 再大，更该先减 r / 减 target_modules，而不是只把 dropout 拧到很大。

    本脚本固定 r=16、alpha=32、全投影模块，只改 dropout。
"""

from _common import ALL_MODULES, fmt, run_lora_tiny


def main():
    """入口：只改 lora_dropout（0 vs 0.05），确认参数量几乎不变、配置能跑通。

    3 步 loss 不能证明谁更好。sft001 用 0.05 是小数据上留一点正则。
    """
    print("固定：r=16, alpha=32, 模块=注意力+MLP")
    print("只改 lora_dropout = 0.0 / 0.05")
    print()

    rows = []
    for drop in (0.0, 0.05):
        print(f"—— lora_dropout={drop} ——")
        rec = run_lora_tiny(
            script="05_compare_dropout",
            tag=f"dropout={drop}",
            r=16,
            lora_alpha=32,
            lora_dropout=drop,
            target_modules=ALL_MODULES,
        )
        rows.append(rec)

    print()
    print(f"{'dropout':<12}{'可训练':<16}{'占比%':<10}{'秒/步':<10}{'loss':<10}{'OOM'}")
    for rec in rows:
        print(
            f"{rec['lora_dropout']:<12}"
            f"{rec['lora_trainable']:<16,}"
            f"{rec['trainable_pct']:<10}"
            f"{fmt(rec['avg_step_s']):<10}"
            f"{fmt(rec['last_loss']):<10}"
            f"{fmt(rec['oom'])}"
        )

    print()
    print("小结论：")
    print("  - 可训练参数数应几乎不变（Dropout 不是新矩阵）。")
    print("  - 极少步的 loss 波动不能证明 0.05 更好，只能证明配置能跑。")
    print("  - sft001 用 0.05：小数据上留一点正则，通常比 0 更稳妥。")
    print()
    print("下一步：python learn/week03/06_compare_targets.py")


if __name__ == "__main__":
    main()
