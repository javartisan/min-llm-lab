"""对比原始模型与微调模型的生成质量，并输出 Markdown 测试报告。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
# 与 sft001.py 对齐：对比 Instruct 底座 vs 微调后模型，更公平
BASE_DIR = ROOT / "models" / "SmolLM2-135M-Instruct"
SFT_DIR = ROOT / "models" / "smollm2-135m-java-sft"
DATA_FILE = ROOT / "data" / "test.jsonl"
REPORT_DIR = ROOT / "reports"

SYSTEM_PROMPT = (
    "你是一个简洁的 Java 助教。请用一两句中文准确回答，不要跑题，不要输出无关代码。"
)

MAX_NEW_TOKENS = 128
MIN_NEW_TOKENS = 8
TEMPERATURE = 0.3  # 评测用偏低温度，输出更稳
TOP_P = 0.9
REPETITION_PENALTY = 1.1


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_model(model_dir: Path):
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"模型目录不存在或不完整: {model_dir}\n"
            f"请先运行 python sft001.py 下载 Instruct 底座并完成微调。"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def build_chat_prompt(tokenizer, user_text: str, add_generation_prompt: bool = True) -> str:
    """把用户问题套进与训练一致的 chat template。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_chat_full(tokenizer, user_text: str, assistant_text: str) -> str:
    """构造完整对话文本，用于计算参考答案 NLL。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _trim_to_complete_sentence(text: str) -> str:
    """若末尾不是完整句子，回退到最近的句末标点，避免报告里出现半截话。"""
    text = text.replace("\ufffd", "").strip()
    if not text:
        return text
    # 未闭合代码块：尽量保留到合理长度，不强行句末裁剪
    if text.count("```") % 2 == 1:
        return text

    def is_sentence_end(s: str, idx: int) -> bool:
        ch = s[idx]
        if ch in "。！？!?\n":
            return True
        if ch == "." and idx > 0 and not s[idx - 1].isalnum():
            return True
        return False

    if is_sentence_end(text, len(text) - 1):
        return text.rstrip()
    last = -1
    for i in range(len(text) - 1, -1, -1):
        if is_sentence_end(text, i):
            last = i
            break
    if last >= 0 and last + 1 >= len(text) // 3:
        return text[: last + 1].strip()
    return text


def _cleanup_completion(text: str) -> str:
    text = text.replace("\ufffd", "").strip()
    if not text:
        return text
    # 未闭合代码块不要按段落截断
    if text.count("```") % 2 == 1:
        return text
    if "\n\n" in text:
        parts = text.split("\n\n")
        head = parts[0].strip()
        if len(head) >= 12:
            if len(parts) > 1 and parts[1].lstrip().startswith("```"):
                code_part = parts[1]
                if code_part.count("```") >= 2:
                    text = head + "\n\n" + code_part
                else:
                    text = head
            else:
                text = head
    return _trim_to_complete_sentence(text)


@torch.no_grad()
def generate(tokenizer, model, user_text: str) -> tuple[str, float]:
    from transformers import StoppingCriteria, StoppingCriteriaList

    # 评测时也必须套 chat template，否则 Instruct/微调模型对不上训练格式
    chat_prompt = build_chat_prompt(tokenizer, user_text, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id

    class SentenceStop(StoppingCriteria):
        """生成足够内容且遇到句末后提前停止，避免撞 max_new_tokens 被截断。"""

        def __call__(self, input_ids, scores, **kwargs):
            gen_len = input_ids.shape[1] - prompt_len
            if gen_len < MIN_NEW_TOKENS:
                return False
            text = tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=True)
            if text.count("```") % 2 == 1:
                return False
            stripped = text.rstrip()
            if len(stripped) < 12:
                return False
            last = stripped[-1]
            if last in "。！？!?":
                return True
            if last == "." and len(stripped) >= 2 and not stripped[-2].isalnum():
                return True
            if "\n\n" in text and len(text.split("\n\n", 1)[0].strip()) >= 12:
                return True
            return False

    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        stopping_criteria=StoppingCriteriaList([SentenceStop()]),
    )
    elapsed = time.perf_counter() - t0

    new_ids = outputs[0][prompt_len:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    return _cleanup_completion(completion), elapsed


@torch.no_grad()
def completion_nll(tokenizer, model, user_text: str, completion: str) -> float:
    """计算参考答案在 chat 格式下的平均 NLL（越低越好）。"""
    prompt_text = build_chat_prompt(tokenizer, user_text, add_generation_prompt=True)
    full_text = build_chat_full(tokenizer, user_text, completion)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    labels = full_ids.clone()
    # 只对 assistant 回答部分计 loss；前面 prompt 部分置为 -100 忽略
    prompt_len = prompt_ids.shape[1]
    labels[:, :prompt_len] = -100
    outputs = model(input_ids=full_ids, labels=labels)
    return float(outputs.loss.item())


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = set(pred)
    ref_tokens = set(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_hit(pred: str, ref: str) -> float:
    """参考答案中英文词 + 中文二元组的命中率（粗粒度相关性）。"""
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

    keys = list(dict.fromkeys(keys))
    if not keys:
        return 0.0
    pred_l = pred.lower()
    hits = sum(1 for k in keys if k in pred_l)
    return hits / len(keys)


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def build_report(
    samples: list[dict],
    base_results: list[dict],
    sft_results: list[dict],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_nll = avg([r["nll"] for r in base_results])
    sft_nll = avg([r["nll"] for r in sft_results])
    base_f1 = avg([r["token_f1"] for r in base_results])
    sft_f1 = avg([r["token_f1"] for r in sft_results])
    base_kw = avg([r["keyword_hit"] for r in base_results])
    sft_kw = avg([r["keyword_hit"] for r in sft_results])
    base_lat = avg([r["latency"] for r in base_results])
    sft_lat = avg([r["latency"] for r in sft_results])

    winner_nll = "微调模型" if sft_nll < base_nll else ("Instruct 底座" if sft_nll > base_nll else "持平")
    winner_f1 = "微调模型" if sft_f1 > base_f1 else ("Instruct 底座" if sft_f1 < base_f1 else "持平")
    winner_kw = "微调模型" if sft_kw > base_kw else ("Instruct 底座" if sft_kw < base_kw else "持平")

    lines = [
        "# Instruct 底座 vs 微调模型 对比测试报告",
        "",
        f"- 生成时间: `{now}`",
        f"- Instruct 底座: `{BASE_DIR}`",
        f"- 微调模型: `{SFT_DIR}`",
        f"- 测试集: `{DATA_FILE}`（{len(samples)} 条）",
        f"- 生成参数: `max_new_tokens={MAX_NEW_TOKENS}`, `temperature={TEMPERATURE}`, `top_p={TOP_P}`, `repetition_penalty={REPETITION_PENALTY}`",
        "",
        "## 指标说明",
        "",
        "| 指标 | 含义 | 方向 |",
        "|---|---|---|",
        "| NLL | 参考答案在模型下的平均负对数似然 | 越低越好 |",
        "| Token F1 | 生成文本与参考答案的字符集合 F1 | 越高越好 |",
        "| Keyword Hit | 参考答案关键片段在生成中的命中率 | 越高越好 |",
        "| Latency | 单条生成耗时（秒） | 越低越好 |",
        "",
        "## 汇总对比",
        "",
        "| 指标 | Instruct 底座 | 微调模型 | 更优方 |",
        "|---|---:|---:|---|",
        f"| NLL | {base_nll:.4f} | {sft_nll:.4f} | {winner_nll} |",
        f"| Token F1 | {base_f1:.4f} | {sft_f1:.4f} | {winner_f1} |",
        f"| Keyword Hit | {base_kw:.4f} | {sft_kw:.4f} | {winner_kw} |",
        f"| Latency (s) | {base_lat:.3f} | {sft_lat:.3f} | - |",
        "",
        "## 逐条生成对比",
        "",
    ]

    for i, sample in enumerate(samples):
        b, s = base_results[i], sft_results[i]
        lines.extend(
            [
                f"### 用例 {i + 1}",
                "",
                f"**Prompt:** {sample['prompt']}",
                "",
                f"**Reference:** {sample['completion']}",
                "",
                "| 模型 | 生成结果 | NLL | Token F1 | Keyword Hit | Latency (s) |",
                "|---|---|---:|---:|---:|---:|",
                f"| Instruct 底座 | {md_escape(b['generation'])} | {b['nll']:.4f} | {b['token_f1']:.4f} | {b['keyword_hit']:.4f} | {b['latency']:.3f} |",
                f"| 微调 | {md_escape(s['generation'])} | {s['nll']:.4f} | {s['token_f1']:.4f} | {s['keyword_hit']:.4f} | {s['latency']:.3f} |",
                "",
            ]
        )

    lines.extend(
        [
            "## 结论",
            "",
            f"- 在参考答案似然（NLL）上，更优: **{winner_nll}**（底座 {base_nll:.4f} vs 微调 {sft_nll:.4f}）。",
            f"- 在字符级重叠（Token F1）上，更优: **{winner_f1}**（底座 {base_f1:.4f} vs 微调 {sft_f1:.4f}）。",
            f"- 在关键词命中上，更优: **{winner_kw}**（底座 {base_kw:.4f} vs 微调 {sft_kw:.4f}）。",
            "- 本测试使用独立测试集（含改写题与相关新题），未使用训练数据；样本量仍较小，结论仅供参考。",
            "",
        ]
    )
    return "\n".join(lines)


def eval_model(name: str, model_dir: Path, samples: list[dict]) -> list[dict]:
    print(f"加载{name}: {model_dir}")
    tokenizer, model = load_model(model_dir)
    results = []
    for i, sample in enumerate(samples, 1):
        prompt = sample["prompt"]
        ref = sample["completion"]
        print(f"  [{name}] {i}/{len(samples)}: {prompt}")
        gen, latency = generate(tokenizer, model, prompt)
        nll = completion_nll(tokenizer, model, prompt, ref)
        results.append(
            {
                "generation": gen,
                "nll": nll,
                "token_f1": token_f1(gen, ref),
                "keyword_hit": keyword_hit(gen, ref),
                "latency": latency,
            }
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    samples = load_samples(DATA_FILE)
    if not samples:
        raise SystemExit(f"测试数据为空: {DATA_FILE}")

    base_results = eval_model("Instruct 底座", BASE_DIR, samples)
    sft_results = eval_model("微调模型", SFT_DIR, samples)

    report = build_report(samples, base_results, sft_results)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"compare_report_{stamp}.md"
    latest_path = REPORT_DIR / "compare_report_latest.md"
    report_path.write_text(report, encoding="utf-8")
    latest_path.write_text(report, encoding="utf-8")

    print("\n===== 汇总 =====")
    print(f"底座 NLL={avg([r['nll'] for r in base_results]):.4f}  "
          f"F1={avg([r['token_f1'] for r in base_results]):.4f}  "
          f"KW={avg([r['keyword_hit'] for r in base_results]):.4f}")
    print(f"微调 NLL={avg([r['nll'] for r in sft_results]):.4f}  "
          f"F1={avg([r['token_f1'] for r in sft_results]):.4f}  "
          f"KW={avg([r['keyword_hit'] for r in sft_results]):.4f}")
    print(f"报告已生成: {report_path}")
    print(f"最新报告:   {latest_path}")


if __name__ == "__main__":
    main()
