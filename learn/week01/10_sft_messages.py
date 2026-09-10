"""Day 5：SFT 数据如何变成对话 messages，以及为什么用 Instruct。

学什么
    Base     = 会续写，不一定会答题
    Instruct = 已经学过「听指令、按角色对话」
    本项目把 {prompt, completion} 转成 system / user / assistant
    chat_template 再把 messages 拼成模型真正看到的字符串

对应阅读
    sft001.py 里的 to_messages
    docs/学习路线/材料/微调学习笔记.md 第 1、2 节

运行
    python learn/week01/10_sft_messages.py
"""

import json

from _common import TRAIN_FILE, load_tokenizer

# 与 sft001.py 保持一致，方便你对照正式训练脚本
SYSTEM_PROMPT = (
    "你是一个简洁的 Java 助教。请用一两句中文准确回答，不要跑题，不要输出无关代码。"
)


def to_messages(prompt: str, completion: str):
    """把一条 jsonl 样本转成对话消息列表。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def main():
    tokenizer, model_dir = load_tokenizer()
    print(f"Tokenizer: {model_dir}")

    print("=" * 60)
    print("【1】Base vs Instruct（先记结论）")
    print("=" * 60)
    print("  Base     : 海量文本预训练，擅长把一段话继续写下去。")
    print("  Instruct : 在 Base 上又学过指令/对话，更会「问一句答一句」。")
    print("  本仓库问答 SFT 用 Instruct，是因为目标是答题，不是写小说。")

    # 读训练集第一条，当作例子
    with TRAIN_FILE.open(encoding="utf-8") as f:
        sample = json.loads(f.readline())
    prompt = sample["prompt"]
    completion = sample["completion"]

    print()
    print("=" * 60)
    print("【2】原始样本（jsonl 里的样子）")
    print("=" * 60)
    print(f"  prompt     = {prompt}")
    print(f"  completion = {completion}")

    messages = to_messages(prompt, completion)
    print()
    print("=" * 60)
    print("【3】转成 messages（sft001.py 就是这样做的）")
    print("=" * 60)
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content']}")

    print()
    print("=" * 60)
    print("【4】chat_template：messages → 模型真正吃进去的文本")
    print("=" * 60)
    # tokenize=False：只要字符串，先不要变成 id，方便人看
    # add_generation_prompt=False：这是「完整对话」（含答案），用于训练
    train_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    print("--- 训练时看到的文本（含 assistant 答案）---")
    print(train_text)

    # 推理时还没有答案，要在末尾留下「assistant 开始说话」的位置
    infer_messages = messages[:2]  # 只留 system + user
    infer_text = tokenizer.apply_chat_template(
        infer_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("--- 推理时看到的文本（等模型接着写 assistant）---")
    print(infer_text)

    print("请观察：")
    print("  - <|im_start|>role  ... <|im_end|>  标出谁在说话")
    print("  - 推理版最后会多出 <|im_start|>assistant ，让模型从这里开始生成")

    print()
    print("小结论：")
    print("  - SFT 用标准问答教模型「遇到这类问题就这样答」。")
    print("  - Instruct + messages + chat_template，才能和对话格式对齐。")


if __name__ == "__main__":
    main()
