# 学习材料索引

本目录是学习计划引用的**全部概念材料**（无需再到仓库其它 docs）。代码仍在仓库根目录的 `learn/`、`scripts/`。

---

## 文档列表

| 文档 | 一句话 | 配合代码 |
|---|---|---|
| [Encoder-Decoder与LLM问答流程.md](./Encoder-Decoder与LLM问答流程.md) | 问答有没有 Encoder、向量从哪来 | `07_forward_logits.py`、`08_embedding.py`、`12_week1_chain.py` |
| [微调学习笔记.md](./微调学习笔记.md) | Base vs Instruct、SFT、DPO、LoRA、训练 8 步 | `09_loss_backward.py`、`10_sft_messages.py`、`scripts/sft001.py` |
| [第1周知识体系与答疑.md](./第1周知识体系与答疑.md) | 概念答疑 + 第 7 节逐脚本代码导读（01～12、`call.py`、`dropout.py`） | `learn/week01/` 全套、`learn/py/` |
| [第1周代码学习.md](./第1周代码学习.md) | week01 **源码 + 运行结果**（侧栏在答疑下面） | `learn/week01/` 01～12、`_common.py` |
| [模型文件说明/README.md](./模型文件说明/README.md) | 模型目录里每个文件干什么 | `01_load_tokenizer.py`、`scripts/print_tokens.py` |
| [01-目录与各文件总览.md](./模型文件说明/01-目录与各文件总览.md) | Base / Instruct / 微调后目录 | 打开 `models/` |
| [02-tokenizer.json结构说明.md](./模型文件说明/02-tokenizer.json结构说明.md) | 词表、BPE、encode ≠ Encoder | `02`～`05` |
| [第2周-BaseAutoModelClass与Auto家族.md](./第2周-BaseAutoModelClass与Auto家族.md) | `_BaseAutoModelClass`、继承图、各 `AutoModelFor*` | `week02/02`、`06`；对照 `sft001.py` |
| [第2周-模型上下文长度涉及的因素.md](./第2周-模型上下文长度涉及的因素.md) | 出厂窗口 vs `num_ctx`、为何难到 1M | Ollama / `config.json` 的 `max_position_embeddings` |

入口：

- [学习大纲.md](../学习大纲.md)  
- [学习计划.md](../学习计划.md)  

---

## 第 1 周：材料 → 脚本

```text
tokenizer.json 说明  →  01～05 加载、切词、BPE、特殊符号
train/test jsonl     →  06 看样本
问答流程 §8/§9      →  07 前向 logits、08 Embedding 查表
微调笔记 §5          →  09 Loss / backward / step（+ dropout.py）
微调笔记 §1/§2       →  10 messages + sft001.py
评测直觉             →  11、eval_compare.py
知识体系第 7 节代码导读  →  12 把 8 个框跑一遍（对照 01～11）
```

## 第 2 周：材料 → 脚本

```text
BaseAutoModelClass 文档 §1～§5  →  Day 2：读 Auto 工厂 + 02_move_to_device
打印 type(model)               →  对照 §7 最小实验
子类任务分组表                 →  Day 6：360M 仍是 CausalLM 入口
设备 / batch 实验              →  01～07（与 Auto 概念并行，互不替代）
上下文长度（出厂 vs 运行时）   →  第2周-模型上下文长度涉及的因素.md
```

一律在仓库根目录执行，例如：

```bash
python learn/week01/01_load_tokenizer.py
python learn/week02/01_probe_device.py
```
