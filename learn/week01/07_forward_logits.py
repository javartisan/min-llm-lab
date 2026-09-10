"""Day 3：Forward（前向）—— input_ids 进模型，得到 logits。

学什么
    一次前向：input_ids → Transformer → logits
    logits 是「还没变成概率的分数」，形状大约是
        [batch_size, 序列长度, 词表大小]
    一次前向通常只帮你预测「下一个 token」，完整回答要循环很多次。

对应阅读
    docs/学习路线/材料/Encoder-Decoder与LLM问答流程.md 第 1 节
    对照仓库里的 scripts/main.py：model(**inputs) 与 logits.shape

运行
    python learn/week01/07_forward_logits.py
"""

import torch

from _common import load_model


def main():
    tokenizer, model, model_dir = load_model()
    model.eval()  # 推理模式：不计算用于训练的梯度，更省内存

    text = "Java interface 是什么？"
    print(f"模型目录: {model_dir}")
    print(f"原文    : {text}")

    # return_tensors="pt" 表示结果做成 PyTorch 张量，才能送进模型
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print()
    print("tokenizer 输出的键:", list(inputs.keys()))
    print("input_ids:", input_ids.tolist())
    print("input_ids.shape:", tuple(input_ids.shape), "  # [batch, seq_len]")

    print()
    print("开始前向（只走一遍网络，不更新参数）...")
    with torch.no_grad():  # 关掉梯度，纯推理
        outputs = model(**inputs)

    logits = outputs.logits
    # 一般是 [1, 序列长度, 49152]
    print("logits.shape:", tuple(logits.shape), "  # [batch, seq_len, vocab_size]")
    print()
    print("怎么读这个形状：")
    print("  batch      = 这次送进去几句话（这里是 1）")
    print("  seq_len    = 这句话有多少个 token")
    print("  vocab_size = 每个位置都要对「下一个词」打 49152 个分数")

    # 取最后一个位置的分数：这就是「紧接着原文，下一个 token 该是谁」的投票
    last_logits = logits[0, -1]  # 形状 [vocab_size]
    next_id = int(torch.argmax(last_logits))
    next_token = tokenizer.decode([next_id])
    print()
    print(f"最后一个位置分数最高的 token id = {next_id}")
    print(f"对应文字 ≈ {next_token!r}")
    print("注意：这只是「下一个碎片」，不是完整答案。")

    print()
    print("用 generate 循环多走几步，才能看到一段话：")
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=12,  # 只生成 12 个新 token，演示即可
            do_sample=False,  # 贪心：每次都选分数最高的，方便复现
        )
    # generate 返回「原文 ids + 新生成 ids」
    new_ids = gen_ids[0, input_ids.shape[1] :]
    print("新生成 ids :", new_ids.tolist())
    print("新生成文字 :", tokenizer.decode(new_ids, skip_special_tokens=True))
    print("如果看起来像乱续写：正常。裸问题没套对话格式，Instruct 模型容易胡编。")

    print()
    print("套上 chat_template 再生成几步（Day 5 会细讲）：")
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    chat_inputs = tokenizer(chat, return_tensors="pt")
    with torch.no_grad():
        chat_gen = model.generate(
            **chat_inputs,
            max_new_tokens=20,
            do_sample=False,
        )
    chat_new = chat_gen[0, chat_inputs["input_ids"].shape[1] :]
    print("对话格式新文字:", tokenizer.decode(chat_new, skip_special_tokens=True))

    print()
    print("小结论：")
    print("  - logits 是分数，还不是概率，更不是最终汉字。")
    print("  - 一次前向 ≠ 生成完整答案；生成是「前向 → 选 1 个 token → 再前向」的循环。")


if __name__ == "__main__":
    main()
