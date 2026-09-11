"""Day 3～5：只改 target_modules（全投影 vs 只注意力）。

学什么
    LoRA 不会自动打在所有线性层上，你要点名：target_modules。
    点得越少，可训练参数越少，通常更省、也更不容易把小数据背下来。
    点得越多，补丁越能改「思考 + 表达」（注意力）和「通道混合」（MLP）。

对应阅读
    scripts/sft001.py 的 target_modules 列表
    学习计划第 3 周：减少模块，例如只留注意力投影

运行
    python learn/week03/06_compare_targets.py

----------------------------------------------------------------------
科普：注意力投影 vs MLP 投影

    Decoder 一层里常见两类线性：

        注意力  q_proj k_proj v_proj o_proj
            决定「看着谁、看出去怎么投」

        MLP     gate_proj up_proj down_proj
            在每个位置上做通道混合（有时叫 FFN）

    sft001 两类都打，表达力够用、小模型也扛得住。
    若只想更省：先只打注意力，看任务还行不行。

    名字必须和模型里的模块名一致。SmolLM2 / Llama 系就是上面这些 *_proj。
    打错名字时 peft 可能警告「没匹配到层」——可训练参数会少得离谱。
"""

from _common import ALL_MODULES, ATTN_MODULES, fmt, run_lora_tiny


def main():
    print("固定：r=16, alpha=32, dropout=0.05")
    print(f"方案 A 全投影     : {ALL_MODULES}")
    print(f"方案 B 只注意力   : {ATTN_MODULES}")
    print()

    plans = [
        ("all_proj", ALL_MODULES),
        ("attn_only", ATTN_MODULES),
    ]
    rows = []
    for tag, mods in plans:
        print(f"—— {tag}  ({len(mods)} 类模块名) ——")
        rec = run_lora_tiny(
            script="06_compare_targets",
            tag=tag,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=mods,
        )
        rows.append(rec)

    print()
    print(f"{'方案':<12}{'模块数':<8}{'可训练':<16}{'占比%':<10}{'秒/步':<10}{'OOM'}")
    for rec in rows:
        print(
            f"{rec['tag']:<12}{rec['n_target_modules']:<8}"
            f"{rec['lora_trainable']:<16,}"
            f"{rec['trainable_pct']:<10}"
            f"{fmt(rec['avg_step_s']):<10}"
            f"{fmt(rec['oom'])}"
        )

    if len(rows) == 2 and rows[0]["lora_trainable"]:
        a, b = rows[0]["lora_trainable"], rows[1]["lora_trainable"]
        print()
        print(f"只打注意力时，可训练参数大约是全投影的 {100.0 * b / a:.1f}%。")

    print()
    print("笔记作业：")
    print("  若内存紧张，你会先减 r，还是先减 target_modules？写一句理由。")
    print()
    print("下一步（可选 1.7B）：python learn/week03/07_try_1p7b.py")
    print("或直接复盘：python learn/week03/08_week3_report.py")


if __name__ == "__main__":
    main()
