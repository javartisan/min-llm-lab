# tokenizer.json 结构说明

本文说明 `tokenizer.json` 内部各主要字段的作用，并澄清常见概念混淆。

对应文件示例：`models/smollm2-135m-java/tokenizer.json`

---

<a id="tok-1"></a>
## 1. 它到底是什么？

`tokenizer.json` **不是**「目标语言领域所有字符的编码表」，也不是 Java 专用字典。

它是预训练时定好的**通用分词引擎配置**，主要包括：

- 通用词表（token ↔ id）
- BPE 切词 / 合并规则
- 特殊符号
- 编码前预处理、解码还原规则

本项目做 SFT/LoRA 时，一般**只改模型权重**，tokenizer 通常从 Instruct 底座原样拷贝，不会按 Java 语料重编一套码。

| 误解 | 实际情况 |
|---|---|
| 领域（Java）全部字符编码 | 否，是通用词表 |
| 每个汉字都单独占一个 ID | 不一定，常被拆成多个子词/字节 token |
| 微调后重新生成了词表 | 通常没有 |

词表大小固定（SmolLM2 约 `vocab_size=49152`）。

---

<a id="tok-2"></a>
## 2. 顶层主要字段

你这份文件顶层大致是：

```text
version
truncation
padding
added_tokens
normalizer
pre_tokenizer
post_processor
decoder
model          ← 核心（vocab + merges）
```

### 2.1 `version`
文件格式版本（如 `1.0`），给加载器识别规范用。

### 2.2 `truncation` / `padding`
默认截断、填充策略。  
为 `null` 时表示不在此文件写死，通常由代码调用时指定。

### 2.3 `added_tokens`
额外特殊 token，例如：

- `<|endoftext|>`
- `<|im_start|>`
- `<|im_end|>`

常见子字段：`id`、`content`、`special`、`lstrip`、`rstrip`。  
对话模型很依赖这些符号。

### 2.4 `normalizer`
切词前的文本规范化（小写、Unicode 正规化等）。  
为 `null` 表示基本不做额外规范化。

### 2.5 `pre_tokenizer`
BPE 之前的「粗切」：按空格/标点/字节级规则先切开。  
你这里常见类型是 `Sequence`（多步组合）。

### 2.6 `post_processor`
分词后的后处理（如 BERT 自动加 `[CLS]`/`[SEP]`）。  
为 `null` 时，对话格式主要靠 `chat_template.jinja`，不靠这里。

### 2.7 `decoder`
把 `input_ids` **还原成可读文本**的规则。  
SmolLM 常见为 `ByteLevel`，负责处理前导空格等细节（如 `Ġ` → 空格）。

### 2.8 `model`（最核心）
真正的分词模型。本项目为 **BPE**：

| 子字段 | 作用 |
|---|---|
| `type` | `BPE` |
| `vocab` | 词表：`token字符串 → id`（约 49152） |
| `merges` | 合并规则列表（约数万条） |
| `unk_token` 等 | 未知词与其它 BPE 开关 |

- **`vocab`**：允许出现哪些 token、对应哪个数字  
- **`merges`**：按什么优先级把小片合成更大 token  

`scripts/print_tokens.py` 打印的 token / input_id，主要来自这里。

---

<a id="tok-3"></a>
## 3. 编码 / 解码流水线

```text
原文
  → normalizer（可选）
  → pre_tokenizer（粗切）
  → model.BPE（merges 合并，查 vocab → input_ids）
  → post_processor（可选）
  → input_ids

反过来：
  input_ids → decoder → 可读文本
```

API 对应关系：

| 方向 | 常用方法 | 结果 |
|---|---|---|
| 文本 → 数字 | `tokenizer.encode(...)` / `tokenize` + 查词表 | `input_ids` |
| 数字 → 文本 | `tokenizer.decode(...)` | 字符串 |

注意：`tokenizer.json` 里通常**没有**名叫 `encoder` 的顶层字段；  
编码能力由 `pre_tokenizer` + `model` 共同完成，解码才单独有 `decoder` 字段。

---

<a id="tok-4"></a>
## 4. 重要概念区分（易混）

| 说法 | 指什么 |
|---|---|
| tokenizer 的 **encode** | 文本 → token → `input_ids` |
| tokenizer 的 **decode** | `input_ids` → 文本 |
| Transformer **Encoder** | 网络结构中的编码层（如 BERT） |
| SmolLM2 | 主要是 **Decoder-only** 因果语言模型，不是 Encoder-Decoder |

初学时最容易把「大模型 Encoder」误当成「tokenizer encode」。  
二者不是一回事。

---

<a id="tok-5"></a>
## 5. token 与 input_id

- **token**：文本片段（子词/字节片等）  
- **input_id**：该片段在词表中的整数编号  

模型计算用的是数字；人查看时再用 decode 还原成文字。  
中文常被拆成多个 token，单个看起来可能像乱码，但整段 decode 后会恢复正常。

---

<a id="tok-6"></a>
## 6. 和同目录其它文件的关系

| 文件 | 关系 |
|---|---|
| `tokenizer.json` | 分词引擎主体（本文件） |
| `tokenizer_config.json` | 怎么用 tokenizer、特殊符号、模板相关配置 |
| `vocab.json` / `merges.txt` | 词表与合并规则的可读/兼容副本 |
| `special_tokens_map.json` | 特殊符号速查 |
| `chat_template.jinja` | 对话消息如何拼成模型输入文本 |

加载时：`AutoTokenizer.from_pretrained(目录)` 会综合读取这些文件。
