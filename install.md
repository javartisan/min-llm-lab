# 环境安装与依赖管理

本文记录本仓库如何安装依赖，以及 Python 里和 Java Maven 类似的工具（含 uv 能否构建项目）。

适用环境：Intel Mac + 本仓库已验证的 `venv`（PyTorch 2.2.2）。

---

## 1. 本仓库推荐安装方式（pip + venv）

这是当前项目正在用的方式，入门足够。

```bash
# 进入仓库根目录
cd /path/to/tiny-llm-lab

# 创建隔离环境（类似「这个项目自己的 classpath」）
python -m venv venv

# 激活环境
source venv/bin/activate

# 按锁定版本安装依赖
pip install -r requirements.txt
```

验证：

```bash
python -c "import torch, transformers, peft, trl; print(torch.__version__)"
```

应打印 `2.2.2`。

之后运行脚本前都先激活 `venv`：

```bash
source venv/bin/activate
python learn/week01/01_load_tokenizer.py
```

退出环境：`deactivate`

---

## 2. Python 有没有类似 Maven 的工具？

有。Python 也能管依赖，只是不像 Maven 那样由一个工具包办「声明 + 下载 + 编译 + 打包」。

Maven 大致在管：依赖版本、下载缓存、打包、生命周期（compile / test / package）。

Python 通常拆成几层：

| 你在 Java 里用的 | Python 常见对应 |
|---|---|
| 依赖声明 `pom.xml` | `requirements.txt` 或 `pyproject.toml` |
| 下载依赖 | `pip` / `uv` / `poetry` |
| 隔离环境 | `venv`（避免和系统 Python、别的项目互相污染） |
| 中央仓库 Maven Central | PyPI |
| 生命周期插件 | 没有官方统一生命周期；运行就是 `python xxx.py` |
| 打包发布 | `uv build` / Poetry / `python -m build` |

常见方案：

| 方案 | 特点 | 适不适合本仓库 |
|---|---|---|
| **pip + venv + requirements.txt** | 最简单 | **适合，当前就是这套** |
| **Poetry / PDM** | 更接近 Maven：一个 `pyproject.toml` 管依赖、环境、打包 | 以后要发布包再用 |
| **uv** | 很快，能管环境、锁版本，也能构建项目 | 想加速安装或迁移时可用 |
| **Conda** | 偏数据科学，还能装非 Python 组件 | 本仓库不必上 |

和 Maven 的两个重要差别：

1. **默认不锁死整棵依赖树。**  
   只写 `transformers==4.57.6` 时，它的下级库版本仍可能浮动。要更稳，可用 `uv.lock` 或 Poetry 的 lock 文件。
2. **没有 `mvn compile` 那种强制生命周期。**  
   Python 脚本直接运行即可，不需要先编译成 class。

---

## 3. uv 可以构建项目吗？

可以。uv 既能创建项目，也能打包构建，比 pip + `requirements.txt` 更接近 Maven。

| Maven | uv |
|---|---|
| `mvn archetype` 建工程 | `uv init` |
| `pom.xml` | `pyproject.toml` |
| `mvn install` 下依赖 | `uv add` / `uv sync` |
| `mvn package` 打 jar | `uv build` 打 wheel / tar.gz |
| `mvn deploy` | `uv publish` |

### 3.1 创建并运行项目

```bash
uv init my-app
cd my-app
uv add torch transformers
uv run python main.py
```

会生成 `pyproject.toml`、虚拟环境，并把精确版本锁到 `uv.lock`。

### 3.2 打包构建（类似 `mvn package`）

```bash
uv build
```

产物在 `dist/`：

- `xxx-0.1.0.tar.gz`：源码包
- `xxx-0.1.0-py3-none-any.whl`：wheel，给别人 `pip install` 用

前提是 `pyproject.toml` 里有项目元数据和构建后端，例如：

```toml
[project]
name = "smollm2-135m-sft"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch==2.2.2", "transformers==4.57.6"]

[build-system]
requires = ["uv_build"]
build-backend = "uv_build"
```

### 3.3 对本仓库意味着什么

日常训练、跑 `learn/week01` 脚本，**不需要** `uv build`。  
那是给「做成可安装的 Python 包、发布到 PyPI」用的。

这个仓库是实验脚本 + 模型微调，不是要发布的库。用 uv 时，更常见的是管环境和依赖：

```bash
uv venv
uv pip install -r requirements.txt

# 以后若迁到 pyproject.toml，则：
uv sync
```

一句话：**uv 能构建项目；本仓库目前用它管环境和依赖就够，不必先打成包。**

---

## 4. 本仓库依赖清单

文件：[`requirements.txt`](./requirements.txt)

| 包 | 用途 |
|---|---|
| `torch==2.2.2` | 训练/推理。Intel Mac 官方最后一版轮子 |
| `transformers` / `tokenizers` | 模型与 Tokenizer |
| `datasets` | `scripts/sft001.py` 读 jsonl |
| `peft` / `trl` / `accelerate` | LoRA + SFT 训练 |
| `safetensors` / `huggingface_hub` | 权重加载与模型下载 |
| `numpy` | torch / datasets 的数值依赖 |
| `markdown` | `docs/学习路线/md2html.py` |

库的职责对照见 [`readme.md`](./readme.md)。

国内下载 Hugging Face 模型不稳定时，可先设置镜像（`scripts/sft001.py` 里已有类似逻辑）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 5. 建议

- **现在：** 继续用 `venv` + `pip install -r requirements.txt`。
- **以后：** 若要多机精确复现、或发布成包，再迁到 uv / Poetry（`pyproject.toml` + lock 文件）。
