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

----------------------------------------------------------------------
科普 1：tokenizer 是变量，为什么还能 tokenizer(...)？

tokenizer 指向一个对象（本项目是 GPT2TokenizerFast）。
Python 里对象只要实现了 __call__，就可以像函数一样写 对象(...)。

    inputs = tokenizer(text, return_tensors="pt")
    # 实际调用：tokenizer.__call__(text, return_tensors="pt")

__call__ 写在父类 PreTrainedTokenizerBase 上，是「送进模型」用的总入口。
同理：model(**inputs) 走的是 model.__call__（里面做前向）。

----------------------------------------------------------------------
科普 2：__call__ / tokenize / encode / convert_tokens_to_ids

四个都在「文本 → 模型输入」这条线上，停的位置不同。
__call__ 不是第四种切词算法，而是 tokenizer(...) 时走的入口。

    原文
      ├─ tokenize                  → 文字碎片 ['Java', 'Ġinterface', ...]
      │       └─ convert_tokens_to_ids → 数字列表 [39570, 8334, ...]
      ├─ encode                    → 数字列表（上面两步合在一起，可加特殊符）
      └─ tokenizer(...) 即 __call__ → 字典：input_ids + attention_mask
                                     还可以变成 PyTorch 张量

    tokenize              输入字符串，输出碎片；给人看怎么切
    convert_tokens_to_ids 输入已经切好的 token 列表，只查表，不会切句
    encode                输入字符串，输出 id 的 Python 列表
    __call__              输入字符串，输出 BatchEncoding 字典，可 padding / 截断 / 张量

关系（不加特殊符号时，本 tokenizer 上内容通常一致）：

    tokenize + convert_tokens_to_ids
        ≈ encode(..., add_special_tokens=False)
        ≈ tokenizer(text)["input_ids"] 里的那串数字

送进模型不要只用 encode 的列表，而用 __call__ 打成的那一包。

----------------------------------------------------------------------
科普 3：__call__ 输出的两个字段

    input_ids       每个位置是哪个 token（词表编号），拿去 Embedding 查向量
    attention_mask  和 input_ids 同样长；1=真实 token，0=padding 填充，注意力应忽略

单句、没有补齐时 mask 全是 1，看起来「没啥用」。
一批句子长短不同、padding=True 时才关键：短句后面会补 pad。
本模型 pad 常和 eos 都是 id=2（<|im_end|>），不给 mask 可能把填充当成正文。

    encode(text)              只有 id 列表
    tokenizer(text)["input_ids"]       同一串 id，多包一层 batch
    tokenizer(text)["attention_mask"]  encode 默认不返回

model(**inputs) = 同时传入这两项：
    input_ids 说「是什么字」
    attention_mask 说「哪些格子是字、哪些是凑长度的空白」
"""

import torch

from _common import load_model


def main():
    tokenizer, model, model_dir = load_model()
    model.eval()  # 推理模式：不计算用于训练的梯度，更省内存

    text = "Java interface 是什么？"
    print(f"模型目录: {model_dir}")
    print(f"原文    : {text}")

    # tokenizer(...) == tokenizer.__call__(...)
    # return_tensors="pt"：做成 PyTorch 张量，才能送进模型
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print()
    print("tokenizer() 即 __call__ 的返回类型:", type(inputs).__name__)
    print("输出的键:", list(inputs.keys()))
    print("input_ids      :", input_ids.tolist(), "  # 每个位置的词表编号")
    print("attention_mask :", attention_mask.tolist(), "  # 1=真实 token，0=padding")
    print("两者形状       :", tuple(input_ids.shape), tuple(attention_mask.shape), "  # [batch, seq_len]")
    print("单句无 padding 时 mask 会全是 1。")

    print()
    print("开始前向（只走一遍网络，不更新参数）...")
    with torch.no_grad():  # 关掉梯度，纯推理
        # **inputs 拆成 model(input_ids=..., attention_mask=...)
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
    print("logits:", logits)
    print("last_logits:", last_logits)
    print("len(last_logits):", len(last_logits))
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
    new_ids = gen_ids[0, input_ids.shape[1]:]
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
    chat_new = chat_gen[0, chat_inputs["input_ids"].shape[1]:]
    print("对话格式新文字:", tokenizer.decode(chat_new, skip_special_tokens=True))

    print()
    print("小结论：")
    print("  - tokenizer(...) 调用的是 __call__，给出 input_ids + attention_mask。")
    print("  - input_ids 是内容；attention_mask 标记哪些位置不是 padding。")
    print("  - logits 是分数，还不是概率，更不是最终汉字。")
    print("  - 一次前向 ≠ 生成完整答案；生成是「前向 → 选 1 个 token → 再前向」的循环。")


if __name__ == "__main__":
    main()
