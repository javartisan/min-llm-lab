# tiny-llm-lab · 小模型微调实验室

持续学习仓库：用小模型把这条链真正跑通——

```text
Pretraining → Base → SFT → LoRA → QLoRA → DPO → Reasoning
```

当前实验底座是 **SmolLM2-135M**（Intel Mac + CPU 也能练手）。后面几周会对照更大模型、MPS、LoRA / QLoRA / DPO，所以仓库名不再锁死在「135M-sft」。

本地文件夹如果还叫 `SmolLM2-135M-sft`，可以自行改名为 `tiny-llm-lab`，不影响代码。

---

## 目录结构

```text
tiny-llm-lab/
├── README.md              # 本说明
├── install.md             # 环境安装、pip / venv / uv 对照 Maven
├── requirements.txt       # 已验证的依赖版本
│
├── learn/                 # 按周的学习脚本（持续往这里加 week02…）
│   └── week01/            # 第 1 周：Tokenizer → Dataset → Forward → Loss → Backward
│
├── scripts/               # 可复用的训练 / 评测入口
│   ├── main.py            # 下载并试跑 Base 模型
│   ├── print_tokens.py    # 打印 token 与 input_ids
│   ├── sft001.py          # Instruct + LoRA 的 SFT
│   └── eval_compare.py    # 底座 vs 微调对比，写 reports/
│
├── data/                  # 训练 / 测试 jsonl（prompt + completion）
├── models/                # 本地下载的模型（不入库）
├── checkpoints/           # 训练中间结果（不入库）
├── reports/               # 评测报告
│
└── docs/学习路线/         # 大纲、计划、阅读材料（可转 HTML）
```

| 目录 | 放什么 | 不放什么 |
|---|---|---|
| `learn/` | 每周拆开的小实验，注释面向初学者 | 正式长时间训练 |
| `scripts/` | 真正训练、评测、下模型 | 概念讲解长文 |
| `docs/` | Markdown 学习材料 | 代码 |
| `data/` | jsonl 语料 | 模型权重 |

---

## 快速开始

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

更完整的依赖管理说明（含 uv / Maven 对照）：见 [install.md](./install.md)。

第 1 周从这里开始：

```bash
python learn/week01/01_load_tokenizer.py
```

阅读顺序：

1. [docs/学习路线/学习大纲.md](./docs/学习路线/学习大纲.md)
2. [docs/学习路线/学习计划.md](./docs/学习路线/学习计划.md)

---

## 本仓库用到的库

| 库 | 核心职责 | 重要程度 |
| --- | --- | ---: |
| **PyTorch** | Tensor、Autograd、神经网络、Optimizer | ⭐⭐⭐⭐⭐ |
| **Transformers** | Transformer 模型、Tokenizer、模型加载 | ⭐⭐⭐⭐⭐ |
| **Datasets** | 数据集加载和处理 | ⭐⭐⭐⭐ |
| **TRL** | SFT、DPO、GRPO 等后训练 | ⭐⭐⭐⭐ |
| **PEFT** | LoRA、Adapter 等参数高效微调 | ⭐⭐⭐⭐ |
| **Accelerate** | CPU/GPU/MPS/分布式训练管理 | ⭐⭐⭐ |
