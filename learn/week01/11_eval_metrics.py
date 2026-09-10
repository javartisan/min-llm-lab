"""Day 6：评测里几个数字大概是什么意思。

学什么
    Reference   = 测试集里的标准答案（completion）
    Token F1    = 预测和标准答案的字面重合程度（本项目按「字符集合」做了简化）
    Keyword Hit = 标准答案里的关键词，预测里命中了多少
    NLL         = 标准答案在模型看来有多「意外」（越低越好）

对应阅读
    eval_compare.py
    reports/compare_report_latest.md

运行
    python learn/week01/11_eval_metrics.py
"""

import json

import torch

from _common import TEST_FILE, load_model


def token_f1(pred: str, ref: str) -> float:
    """与 eval_compare.py 相同的简化版：按字符集合算 F1。

    这不是严格语言学上的分词 F1，只是一个「字面像不像」的粗指标。
    """
    pred_set = set(pred)
    ref_set = set(ref)
    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0
    overlap = len(pred_set & ref_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    with TEST_FILE.open(encoding="utf-8") as f:
        sample = json.loads(f.readline())
    question = sample["prompt"]
    reference = sample["completion"]  # 标准答案

    print("=" * 60)
    print("【1】Reference 是什么？")
    print("=" * 60)
    print(f"问题      : {question}")
    print(f"Reference : {reference}")
    print("Reference 来自 test.jsonl 的 completion，不是模型自己生成的。")

    print()
    print("=" * 60)
    print("【2】Token F1：字面重合")
    print("=" * 60)
    good = reference  # 假如模型一字不差答对了
    bad = "今天天气很好，我们去吃饭吧。"  # 完全跑题
    print(f"完全相同的预测 F1 = {token_f1(good, reference):.3f}   （应接近 1）")
    print(f"完全跑题的预测 F1 = {token_f1(bad, reference):.3f}   （应接近 0）")
    print("F1 高 ≠ 一定懂了，只说明用字和标准答案像。")

    print()
    print("=" * 60)
    print("【3】NLL：标准答案在模型眼里有多「顺」")
    print("=" * 60)
    print("正在加载模型，对这一条标准答案算一次 loss（CPU 可能要几秒）...")
    tokenizer, model, model_dir = load_model()
    model.eval()
    print(f"模型: {model_dir}")

    # 为了好懂，这里不用完整 chat_template，直接把「问题+答案」拼成一段
    # 正式评测见 eval_compare.py 的 completion_nll（会套 chat 格式，且只对答案计 loss）
    text = question + reference
    inputs = tokenizer(text, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        nll = model(input_ids=inputs["input_ids"], labels=labels).loss.item()
    print(f"整段文本的平均 NLL ≈ {nll:.4f}")
    print("读法：数字越小，模型越「觉得这段话像自己会写的」。")
    print("微调后，Java 问答的 NLL 通常会下降（更贴标准答案）。")

    print()
    print("完整对比请看：")
    print("  python scripts/eval_compare.py")
    print("  reports/compare_report_latest.md")

    print()
    print("小结论：")
    print("  - Reference 是标准答案。")
    print("  - F1 / 关键词命中看「像不像」；NLL 看「模型内部有多确信」。")


if __name__ == "__main__":
    main()
