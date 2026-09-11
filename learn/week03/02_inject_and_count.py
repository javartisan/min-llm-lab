"""Day 1：把 LoRA 真正打进 135M，数一数谁在训练。

学什么
    peft.get_peft_model 会：冻住底座、按 LoraConfig 在指定线性层旁挂 A/B。
    对照 scripts/sft001.py 的 build_lora_config：默认 r=16、alpha=32、dropout=0.05。

对应阅读
    scripts/sft001.py 的 build_lora_config
    docs/学习路线/材料/微调学习笔记.md 第 6 节

运行
    python learn/week03/02_inject_and_count.py

需要本地 135M（与 week01/02 相同目录）。

----------------------------------------------------------------------
科普：注入之后，requires_grad 发生了什么

    加载后、未挂 LoRA：几乎所有参数 requires_grad=True
        → 若这时 trainer.train()，就是全量微调。

    get_peft_model 之后：
        底座权重 requires_grad=False（冻住）
        新加的 lora_A / lora_B requires_grad=True

    前向仍然走完整网络（底座 + 旁路），反传却几乎只更新补丁。
    所以省的是「可训练参数」和优化器状态，不是把模型变小后推理。
    推理时若 merge，才会把补丁加回 W，变成一份普通完整权重。
"""

from _common import (
    ALL_MODULES,
    append_run,
    count_params,
    find_model_dir,
    fmt,
    free_model,
    inject_lora,
    load_causal_lm,
    load_texts,
    lora_param_names,
    make_batch,
    make_lora_config,
    pct,
    pick_device,
    timed_train_steps,
)


def main():
    device = pick_device()
    model_dir = find_model_dir("135m")
    print(f"设备 : {device}   模型 : {model_dir.name}")
    print("配置对齐 sft001：r=16, lora_alpha=32, lora_dropout=0.05")
    print(f"target_modules = {ALL_MODULES}")

    tokenizer, model = load_causal_lm(model_dir, device)
    total0, train0 = count_params(model)

    print()
    print("=" * 60)
    print("【1】挂 LoRA 之前（相当于全量微调）")
    print("=" * 60)
    print(f"  总参数       {total0:,}")
    print(f"  可训练参数   {train0:,}  ({pct(train0, total0):.2f}%)")
    print("  这时几乎整个网络都在等着被改。")

    config = make_lora_config()
    model = inject_lora(model, config)
    total1, train1 = count_params(model)

    print()
    print("=" * 60)
    print("【2】挂 LoRA 之后")
    print("=" * 60)
    print(f"  总参数       {total1:,}   （比原来略多：多出来的就是 A/B）")
    print(f"  可训练参数   {train1:,}  ({pct(train1, total1):.3f}%)")
    print(f"  相对全量     {100.0 * train1 / train0:.3f}% 的参数在更新")
    if hasattr(model, "print_trainable_parameters"):
        print()
        print("  peft 自己的汇总：")
        model.print_trainable_parameters()

    names = lora_param_names(model)
    print()
    print("  LoRA 参数名长什么样（前几个）：")
    for n in names:
        print(f"    {n}")
    if names:
        print("  名字里有 lora_A / lora_B，层名里能看到 q_proj 等 target_modules。")

    print()
    print("=" * 60)
    print("【3】极少步：确认冻底座时 loss 仍能往下走")
    print("=" * 60)
    batch = make_batch(tokenizer, load_texts(1), device, max_length=64)
    stats = timed_train_steps(model, batch, steps=3)
    print(
        f"  秒/步={fmt(stats['avg_step_s'])}  last_loss={fmt(stats['last_loss'])}"
        f"  RSS={fmt(stats['rss_mb'], 1)} MB  OOM={fmt(stats['oom'])}"
    )
    print("  这 3 步不是认真微调，只证明：只更新补丁时，训练循环仍然成立。")

    append_run(
        {
            "script": "02_inject_and_count",
            "device": str(device),
            "model": model_dir.name,
            "full_params": total0,
            "full_trainable": train0,
            "lora_total_params": total1,
            "lora_trainable": train1,
            "trainable_pct": round(pct(train1, total1), 4),
            **stats,
        }
    )
    free_model(model)

    print()
    print("小结论：")
    print("  - LoRA 没有删掉大模型，只是不让大模型的数字被改。")
    print("  - 可训练参数从「一亿级」掉到「百万级」量级（以你屏幕上的数字为准）。")
    print("  - 正式长时间训练仍走 scripts/sft001.py，不要用本脚本当生产训练。")
    print()
    print("下一步：python learn/week03/03_adapter_files.py")


if __name__ == "__main__":
    main()
