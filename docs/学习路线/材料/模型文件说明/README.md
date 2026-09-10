# 模型文件说明（索引）

本文件夹专门记录：**模型目录里各个文件是什么、干什么用**。

适合对照 `models/`、`checkpoints/` 阅读。

---

## 文档列表

| 文件 | 内容 |
|---|---|
| [01-目录与各文件总览.md](./01-目录与各文件总览.md) | 各模型文件夹含义；从 [第 1 节](./01-目录与各文件总览.md#files-1) 开始 |
| [02-tokenizer.json结构说明.md](./02-tokenizer.json结构说明.md) | tokenizer 内部字段；从 [第 1 节](./02-tokenizer.json结构说明.md#tok-1) / [第 4 节易混概念](./02-tokenizer.json结构说明.md#tok-4) 开始 |

---

## 建议阅读顺序

1. 先看 **01**：建立「目录里每个文件」的整体地图  
2. 再看 **02**：深入 `tokenizer.json`（你正在打开的大文件）  
3. 配合运行：`python scripts/print_tokens.py` 或 `python learn/week01/02_token_and_id.py` 观察 token 与 input_ids  

---

## 相关其它笔记

- 同级材料：`../微调学习笔记.md`（SFT / DPO / Instruct / 训练流程）
- 同级材料：`../Encoder-Decoder与LLM问答流程.md`
- 学习计划：`../../学习计划.md`
