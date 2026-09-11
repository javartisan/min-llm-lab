"""Day 6：评测里几个数字大概是什么意思。

学什么
    Reference   = 测试集里的标准答案（completion），是评测输入，不是分数
    Token F1    = 预测和标准答案的字面重合（本项目按「字符集合」简化）
    Keyword Hit = 标准答案里抽出的关键词，预测里命中了多少（本项目自制）
    NLL         = Negative Log-Likelihood，负对数似然（越低越好）
    PPL         = Perplexity，困惑度；PPL = exp(NLL)

对应阅读
    eval_compare.py
    reports/compare_report_latest.md
    docs/学习路线/材料/第1周知识体系与答疑.md 第 5 节

运行
    python learn/week01/11_eval_metrics.py

----------------------------------------------------------------------
科普：这几个是不是行业常用指标？

先分清：Reference 不是指标，是「拿什么当标准答案」。
有了 Reference，才能算 F1 / Keyword / NLL 这类自动分数。

    名称            行业常不常用                         本仓库怎么实现
    Reference       人人都用（也叫 gold / 参考答案）     test.jsonl 的 completion
    Token F1        「F1」极常用；本实现是简化版         把字符串当成「不重复的字符集合」
    Keyword Hit     不是标准名称，是领域自制启发式       英文词 + 中文二字片，看预测里有没有
    NLL             是（语言模型核心指标）               参考答案的平均负对数似然；正式脚本只对答案计

行业里更常见的「亲戚」：

    和字面重合一类
        Exact Match（EM）   生成和参考是否完全相同（SQuAD 常用）
        Token F1            按分词后的词袋重叠算 P/R/F1（SQuAD 的 F1 带词频，不是 set）
        BLEU                机器翻译：n-gram 精确率
        ROUGE-L             摘要：最长公共子序列，和 Keyword Hit 更像一类
    和「模型觉得顺不顺」一类
        NLL / Cross-Entropy 训练 loss 往往就是平均 NLL
        PPL = Perplexity（困惑度）  exp(NLL)，越低越「不意外」
    和开放生成 / 产品一类（本周脚本没有算，知道即可）
        BERTScore           用向量相似度，不要求用字一样
        Pass@k              代码题：k 次生成里有没有通过测试
        人工 / Arena        人打分或 A vs B；聊天质量最终仍靠这个
        延迟 / 成本         eval_compare.py 已记 Latency

结论：NLL 是正统语言模型指标；F1 公式正统，但本项目用「字符集合」做了教学简化；
Keyword Hit 是为了小数据 Java 短答案加的关键词覆盖率，厂里不会单独叫这个名字，
思路接近「要点召回 / ROUGE」。开放聊天不能只看这三项，字面重合高 ≠ 一定懂了。

----------------------------------------------------------------------
科普：每个指标怎么算（手算示例，和下面 main 打印的数字一致）

【Reference】
    问题:  Java 接口能不能用 new 创建对象？
    参考:  不能，interface 不能直接被实例化。
    它本身没有「得分」，后面所有公式都拿它当右边的标准。

【Token F1】（本项目：字符去重后的集合 F1）
    参考 ref  = "不能实例化"
    预测 pred = "不能直接实例化"

    ref 集合  = {不, 能, 实, 例, 化}           大小 5
    pred 集合 = {不, 能, 直, 接, 实, 例, 化}   大小 7
    交集      = {不, 能, 实, 例, 化}           大小 5

    Precision（预测里有多少字出现在参考中）= 5/7 ≈ 0.714
    Recall   （参考里有多少字出现在预测中）= 5/5 = 1.000
    F1 = 2 * P * R / (P + R) ≈ 0.833

    注意：顺序、重复次数都丢掉了。
    「实例化不能」和「不能实例化」F1 一样；这就是简化的代价。
    行业 SQuAD F1 会按 token 计数（多重集合），并常先规范化标点。

【Keyword Hit】（本项目：参考里抽关键词，看预测覆盖率）
    参考: "interface 不能实例化"
    预测: "接口不能被实例化"

    抽关键词规则（与 eval_compare.keyword_hit 相同）：
      连续英文（长度≥2）→ 一个词，转小写：interface
      连续中文（长度≥2）→ 切成二字片：不能、能实、实例、例化
    关键词 = [interface, 不能, 能实, 实例, 例化]   共 5 个
    预测里出现：不能、实例、例化（3 个）；interface、能实 没有
    Keyword Hit = 3/5 = 0.600

【NLL】（负对数似然，越低越好）
    假设参考答案被切成 3 个 token，模型给出的概率是：
      P(t1)=0.50, P(t2)=0.40, P(t3)=0.20

    单个 token 的 NLL = -ln(P)
      -ln(0.50) ≈ 0.693
      -ln(0.40) ≈ 0.916
      -ln(0.20) ≈ 1.609
    平均 NLL = (0.693+0.916+1.609)/3 ≈ 1.073
    PPL = Perplexity（困惑度）= exp(NLL) = exp(1.073) ≈ 2.92
          可以理解为「平均要在约 3 个候选里猜下一个 token」

    训练时的 cross-entropy loss 就是这种平均 NLL。
    11 脚本为了好懂，把「问题+答案」整段算 loss；
    eval_compare.completion_nll 会套 chat_template，且只对 assistant 答案计 loss。
"""

from __future__ import annotations

import json
import math

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


def extract_keywords(ref: str) -> list[str]:
    """抽出 Keyword Hit 用的关键词列表，便于打印手算过程。"""
    keys: list[str] = []
    buf: list[str] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        token = "".join(buf)
        if token.isascii() and len(token) >= 2:
            keys.append(token.lower())
        elif not token.isascii() and len(token) >= 2:
            for i in range(len(token) - 1):
                keys.append(token[i : i + 2])
        buf = []

    for ch in ref:
        if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"):
            buf.append(ch)
        else:
            flush()
    flush()
    return list(dict.fromkeys(keys))


def keyword_hit(pred: str, ref: str) -> float:
    """与 eval_compare.py 相同：参考答案中英文词 + 中文二元组的命中率。"""
    keys = extract_keywords(ref)
    if not keys:
        return 0.0
    pred_l = pred.lower()
    hits = sum(1 for k in keys if k in pred_l)
    return hits / len(keys)


def show_token_f1_example(pred: str, ref: str) -> None:
    pred_set = set(pred)
    ref_set = set(ref)
    overlap = pred_set & ref_set
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(ref_set)
    f1 = token_f1(pred, ref)
    print(f"  参考 ref  = {ref!r}")
    print(f"  预测 pred = {pred!r}")
    print(f"  ref 集合  = {sorted(ref_set)}   大小 {len(ref_set)}")
    print(f"  pred 集合 = {sorted(pred_set)}   大小 {len(pred_set)}")
    print(f"  交集      = {sorted(overlap)}   大小 {len(overlap)}")
    print(f"  Precision = {len(overlap)}/{len(pred_set)} = {precision:.3f}  （预测里有多少字来自参考）")
    print(f"  Recall    = {len(overlap)}/{len(ref_set)} = {recall:.3f}  （参考里有多少字出现在预测）")
    print(f"  F1        = 2PR/(P+R) = {f1:.3f}")


def show_keyword_example(pred: str, ref: str) -> None:
    keys = extract_keywords(ref)
    pred_l = pred.lower()
    hit_list = [k for k in keys if k in pred_l]
    miss_list = [k for k in keys if k not in pred_l]
    score = keyword_hit(pred, ref)
    print(f"  参考 ref  = {ref!r}")
    print(f"  预测 pred = {pred!r}")
    print(f"  抽出关键词 = {keys}   共 {len(keys)} 个")
    print(f"  命中       = {hit_list}")
    print(f"  未命中     = {miss_list}")
    print(f"  Keyword Hit = {len(hit_list)}/{len(keys)} = {score:.3f}")


def show_toy_nll() -> None:
    probs = [0.50, 0.40, 0.20]
    token_nll = [-math.log(p) for p in probs]
    avg_nll = sum(token_nll) / len(token_nll)
    ppl = math.exp(avg_nll)
    print("  假设参考答案 3 个 token 的概率 P =", probs)
    print("  每个 token 的 -ln(P) =", [round(x, 3) for x in token_nll])
    print(f"  平均 NLL = {avg_nll:.3f}   （Negative Log-Likelihood，负对数似然，越低越好）")
    print(f"  PPL = exp(NLL) = {ppl:.2f}")
    print("  PPL 是 Perplexity（困惑度）的缩写。")
    print("  读法：NLL 是对数空间里的意外程度；PPL 还原成「平均要在几个候选里猜」。")


def main():
    with TEST_FILE.open(encoding="utf-8") as f:
        sample = json.loads(f.readline())
    question = sample["prompt"]
    reference = sample["completion"]  # 标准答案

    print("=" * 60)
    print("【0】是不是行业常用？一句话")
    print("=" * 60)
    print("  Reference    : 不是分数，是标准答案（行业都用这个输入）")
    print("  Token F1     : F1 常用；本项目是「字符去重集合」简化版")
    print("  Keyword Hit  : 不是标准名称，是自制关键词覆盖率")
    print("  NLL          : Negative Log-Likelihood，负对数似然；是核心指标")
    print("  PPL          : Perplexity（困惑度）的缩写；PPL = exp(NLL)")
    print("  行业还会用   : EM、BLEU、ROUGE、BERTScore、Pass@k、人工对比、延迟")

    print()
    print("=" * 60)
    print("【1】Reference 是什么？")
    print("=" * 60)
    print(f"问题      : {question}")
    print(f"Reference : {reference}")
    print("Reference 来自 test.jsonl 的 completion，不是模型自己生成的。")
    print("没有它，F1 / Keyword / NLL 都没法定「对不对」。")

    print()
    print("=" * 60)
    print("【2】Token F1：手算示例（字符集合）")
    print("=" * 60)
    show_token_f1_example("不能直接实例化", "不能实例化")
    print()
    good = reference
    bad = "今天天气很好，我们去吃饭吧。"
    print(f"  用真实参考、完全相同的预测 F1 = {token_f1(good, reference):.3f}   （应接近 1）")
    print(f"  用真实参考、完全跑题的预测 F1 = {token_f1(bad, reference):.3f}   （应接近 0）")
    print("  F1 高 ≠ 一定懂了，只说明用字和标准答案像。顺序被忽略。")

    print()
    print("=" * 60)
    print("【3】Keyword Hit：手算示例")
    print("=" * 60)
    show_keyword_example("接口不能被实例化", "interface 不能实例化")
    print("  英文整词、中文二字片；预测里出现就算命中。")
    print("  近亲是 ROUGE / 要点召回，不是公开榜单上的标准名字。")

    print()
    print("=" * 60)
    print("【4】NLL：手算示例（先不加载模型）")
    print("=" * 60)
    show_toy_nll()

    print()
    print("=" * 60)
    print("【5】NLL：用真实模型对第一条测试样本算一次")
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
    print(f"整段文本的平均 NLL ≈ {nll:.4f}    PPL(Perplexity 困惑度) ≈ {math.exp(nll):.2f}")
    print("PPL 是 Perplexity 的缩写；PPL = exp(NLL)。两者都是越小越好。")
    print("读法：数字越小，模型越「觉得这段话像自己会写的」。")
    print("微调后，Java 问答的 NLL 通常会下降（更贴标准答案）。")
    print("注意：正式脚本只对 assistant 答案计 NLL，数值会和这里的「整段」不同。")

    print()
    print("完整对比请看：")
    print("  python scripts/eval_compare.py")
    print("  reports/compare_report_latest.md")

    print()
    print("小结论：")
    print("  - Reference 是标准答案，不是指标。")
    print("  - F1 / Keyword 看「像不像」；NLL 看「模型内部有多确信」。")
    print("  - PPL = Perplexity（困惑度）= exp(NLL)，和 NLL 说的是同一件事。")
    print("  - 只有 NLL 是正统 LM 指标；本仓库 F1/Keyword 是小数据短答案的简化尺。")


if __name__ == "__main__":
    main()
