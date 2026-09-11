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

----------------------------------------------------------------------
科普：apply_chat_template 在干什么？

messages 只是 Python 列表（role + content）。模型不认识这个列表，
只认识「按它预训练时见过的格式拼出来的一整段字符串」。

apply_chat_template = 用模型目录里的 chat_template.jinja（ChatML）
把消息列表渲染成那段字符串。人手拼 <|im_start|> 容易漏符号；
走模板才能和 Instruct 训练时的格式对齐。

    messages（人看的结构）
        ↓  apply_chat_template
    一段带 <|im_start|>role ... <|im_end|> 的文本（模型看的格式）
        ↓  再 tokenizer(...)  （本脚本先不这一步）
    input_ids

本脚本用到的三个参数：

    第 1 个位置参数 conversation / messages
        消息列表。每条是 {"role": "...", "content": "..."}。
        role 只能是模板认识的：system / user / assistant。
        训练：三轮都给（含 assistant 标准答案）。
        推理：只给已经发生的轮次（system + user），答案还没写出来。

    tokenize=False
        False：返回 str，方便 print 对照。
        True（默认）：内部再 encode，直接返回 input_ids。
        学习阶段先看字符串；真正进模型时再 tokenizer(text, return_tensors="pt")。

    add_generation_prompt
        False：只渲染你传入的消息，到最后一条的 <|im_end|> 为止。
                训练用：文本里已经包含 assistant 答案，当标签学。
        True ：在末尾再追加「assistant 开始说话」的提示
                （本模型是 <|im_start|>assistant 换行）。
                推理用：告诉模型「从这里开始生成回复」，
                不要再续写 user，也不要先输出 <|im_end|> 把自己结束掉。

其它常用参数（本脚本未传，知道即可）：

    return_tensors="pt"   tokenize=True 时，顺便做成张量
    padding / truncation / max_length   批处理补齐、截断
    continue_final_message=True         最后一条消息还没说完，不要先加 <|im_end|>
    chat_template=...                   临时换一套模板（一般不用，用模型自带的）

训练 vs 推理对照（本脚本会打印两段）：

    训练  messages 含 assistant + add_generation_prompt=False
          → 完整对话，含标准答案
    推理  messages 只有 system/user + add_generation_prompt=True
          → 停在 <|im_start|>assistant 之后，等 generate 续写
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
    # apply_chat_template：按 chat_template.jinja 把 messages 拼成 ChatML 字符串。
    # 不是再训练一遍，只是「填模板」。
    train_text = tokenizer.apply_chat_template(
        # conversation：完整三轮（system / user / assistant），assistant 里是标准答案
        messages,
        # tokenize：False=返回字符串；True=直接返回 token id（默认 True）
        tokenize=False,
        # add_generation_prompt：False=渲染到最后一条消息结束为止，不再追加
        # 「assistant 开口」标记。训练需要整段含答案的文本当标签。
        add_generation_prompt=False,
    )
    print("--- 训练时看到的文本（含 assistant 答案）---")
    print(train_text)

    # 推理时还没有答案：不能把 completion 塞进 messages，否则等于把标准答案泄露给模型。
    infer_messages = messages[:2]  # 只留 system + user，去掉 assistant
    infer_text = tokenizer.apply_chat_template(
        # conversation：只有已经发生的轮次，没有答案
        infer_messages,
        tokenize=False,  # 同样先看字符串；真正生成前再 tokenizer(infer_text, return_tensors="pt")
        # True=在末尾追加 <|im_start|>assistant 和换行，把光标放到「该模型说话」的位置。
        # generate() 从这里往后续写；若仍 False，模型可能接着写 user 或立刻结束。
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
