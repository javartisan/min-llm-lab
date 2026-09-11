"""第 3 周共用工具：注入 LoRA、数可训练参数、跑极少步、记实验。

本周一句话：冻住大模型权重，只训练旁路小补丁（A、B 矩阵）。
正式 LoRA 训练：scripts/sft001.py。
正式全量对照：learn/week03/09_full_sft_135m.py（不传 peft_config）。
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import torch.backends.mps as _mps

if not hasattr(_mps, "is_macos_or_newer"):

    def _is_macos_or_newer(major: int, minor: int = 0) -> bool:
        """兼容补丁：新版 transformers 会问「是不是够新的 macOS」。

        Intel Mac 上的 PyTorch 2.2.2 没有这个函数，不补会在加载模型时直接报错。
        这里用旧 API is_macos13_or_newer 拼出同样的是/否。
        """
        if major > 13:
            return False
        if major == 13:
            return _mps.is_macos13_or_newer(minor)
        return True

    _mps.is_macos_or_newer = _is_macos_or_newer

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_FILE = ROOT / "data" / "train.jsonl"
REPORTS_DIR = ROOT / "reports"
RUNS_FILE = REPORTS_DIR / "week03_runs.jsonl"
CHECKPOINT_DIR = ROOT / "checkpoints" / "smollm2-135m-java-lora"
MERGED_DIR = ROOT / "models" / "smollm2-135m-java"
HF_1P7 = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

MODELS_135 = [
    ROOT / "models" / "SmolLM2-135M-Instruct",
    ROOT / "models" / "SmolLM2-135M",
    ROOT / "models" / "smollm2-135m-java-sft",
    ROOT / "models" / "smollm2-135m-java",
]
MODELS_1P7 = [
    ROOT / "models" / "SmolLM2-1.7B-Instruct",
    ROOT / "models" / "SmolLM2-1.7B",
]

# 与 sft001.py 对齐，方便对照正式脚本
ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ALL_MODULES = ATTN_MODULES + MLP_MODULES


def patch_hf_mirror() -> None:
    """国内访问 Hugging Face 不稳定时，默认改走 hf-mirror 镜像。

    setdefault：如果环境变量里已经设了 HF_ENDPOINT，就尊重你的设置，不覆盖。
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def mps_available() -> bool:
    """本机现在能不能用苹果 GPU（MPS）。

    要同时满足：这份 PyTorch 带了 mps 后端，并且当前机器 is_available()==True。
    Intel Mac + 旧轮子上经常是 False，后面就会退回 CPU。
    """
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def pick_device() -> torch.device:
    """选出本周脚本默认把模型和张量送到哪：有 MPS 用 MPS，否则 CPU。"""
    return torch.device("mps") if mps_available() else torch.device("cpu")


def rss_mb() -> float:
    """当前进程内存高水位，单位 MB（粗，只能看趋势）。

    macOS 的 ru_maxrss 是字节；Linux 一般是 KB。不能用来证明「小 batch 更省」，
    同进程里这个值往往只升不降。
    """
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024.0


def find_model_dir(size: str = "135m") -> Path:
    """在 models/ 里找一份能用的本地权重目录（必须有 config.json）。

    size 以 1.7 / 1p7 开头就找 1.7B，否则找 135M。
    按 MODELS_135 / MODELS_1P7 的顺序取第一个命中的，优先 Instruct。
    找不到就抛 FileNotFoundError，并提示怎么下载。
    """
    cands = MODELS_1P7 if size.lower().startswith("1.7") or size.lower().startswith("1p7") else MODELS_135
    for path in cands:
        if (path / "config.json").exists():
            return path
    kind = "1.7B" if cands is MODELS_1P7 else "135M"
    extra = " 可试：python learn/week03/07_try_1p7b.py --download" if kind == "1.7B" else ""
    raise FileNotFoundError(f"本地未找到 {kind}。{extra}")


def load_causal_lm(model_dir: Path, device: torch.device):
    """从本地目录加载 tokenizer + CausalLM，并搬到指定设备。

    local_files_only=True：禁止悄悄联网。没有 pad_token 时用 eos 顶上，
    否则 padding 组 batch 会报错。返回 (tokenizer, model)。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    return tokenizer, model


def free_model(model) -> None:
    """删掉模型引用并尽量清缓存，方便同一脚本里连续跑两组对照。

    Python 不会立刻把内存还给系统；MPS 上再调 empty_cache 只是尽量腾显存。
    """
    del model
    gc.collect()
    if mps_available() and hasattr(torch, "mps"):
        fn = getattr(torch.mps, "empty_cache", None)
        if callable(fn):
            fn()


def count_params(model) -> tuple[int, int]:
    """数参数：(总个数, 可训练个数)。

    numel() 是一张张量里有多少个数。requires_grad=True 才会被 optimizer 更新。
    挂 LoRA 前后各数一次，就能看出「全量 vs 只训补丁」。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def pct(trainable: int, total: int) -> float:
    """可训练参数占总参数的百分比。total=0 时返回 0，避免除零。"""
    return 100.0 * trainable / total if total else 0.0


def make_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """组装一份 LoraConfig，默认值和 scripts/sft001.py 对齐。

    r           秩，旁路有多厚
    lora_alpha  与 r 一起决定缩放 alpha/r
    lora_dropout 训练时随机丢掉一部分旁路
    target_modules 打在哪些线性层；None 则用注意力+MLP 全投影
    task_type=CAUSAL_LM：从左到右生成的语言模型
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules or ALL_MODULES),
    )


def inject_lora(model, config: LoraConfig):
    """把 LoRA 旁路打进模型：冻住原权重，只让新的 A/B 可训练。

    底层是 peft.get_peft_model。返回 PeftModel，前向仍走「底座 + 旁路」。

    Peft 是 "Parameter-Efficient Fine-Tuning" 的缩写，指一种在大型预训练模型（如大语言模型）上进行微调时，只调整少量参数而非全部参数的技术。
    """
    return get_peft_model(model, config)


def load_texts(n: int) -> list[str]:
    """从 data/train.jsonl 取 n 条，把 prompt+completion 拼成短文本。

    只为组一个小 batch 做极少步练习，不是认真的 SFT 数据管线。
    条数不够就循环重复已有样本。
    """
    texts: list[str] = []
    with TRAIN_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(str(row.get("prompt", "")) + str(row.get("completion", "")))
            if len(texts) >= n:
                break
    if not texts:
        raise FileNotFoundError(f"训练数据为空：{TRAIN_FILE}")
    while len(texts) < n:
        texts.append(texts[len(texts) % max(len(texts), 1)])
    return texts


def make_batch(tokenizer, texts: list[str], device: torch.device, max_length: int = 64) -> dict:
    """把若干字符串编成模型能吃的一个 batch。

    padding/truncation 后得到 input_ids、attention_mask，并搬到 device。
    labels 先复制 input_ids：练习脚本对整段算 loss，正式 SFT 往往只对 assistant 计损失。
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    batch = {k: v.to(device) for k, v in enc.items()}
    batch["labels"] = batch["input_ids"].clone()
    return batch


def timed_train_steps(model, batch: dict, *, steps: int = 3, lr: float = 1e-4) -> dict:
    """跑极少步：Forward → Loss → backward → step，记下耗时 / loss / 是否 OOM。

    只优化 requires_grad=True 的参数（挂 LoRA 后就是补丁）。
    先空跑一步清掉编译/缓存干扰，再计时。不是完整 SFT，不要看这几步比效果。
    """
    # 切到训练模式：Dropout 等按「训练时」规则工作。它本身不改权重。
    model.train()
    # 只收集允许被梯度更新的张量。挂 LoRA 后这里几乎只剩 A/B，底座冻住进不来。
    params = [p for p in model.parameters() if p.requires_grad]
    # 优化器登记这些参数。SGD（Stochastic Gradient Descent 随机梯度下降） 不负责前向，只在 step() 时用 .grad 改参数值。
    optimizer = torch.optim.SGD(params, lr=lr)
    # 参数在哪块设备上，后面 MPS 同步就跟这块走。
    device = next(model.parameters()).device
    # 清掉可能残留的旧梯度。set_to_none=True 比填 0 更省一点。
    optimizer.zero_grad(set_to_none=True)
    # 热身：先完整走一遍前向+反传，把首次编译/缓存的开销甩掉，不计入后面计时。
    (model(**batch).loss).backward()
    # 热身产生的梯度不要留下来，正式计时步从零梯度开始。
    optimizer.zero_grad(set_to_none=True)

    times: list[float] = []  # 每一步墙钟耗时（秒）
    last_loss = None  # 最后一步的标量 loss，给屏幕/jsonl 看
    oom = False  # 是否在循环里撞上显存/内存不够
    err = ""  # OOM 时记下原始报错，其它异常继续往上抛
    try:
        # 正式计时的训练步。每步都用同一个 batch，只为测速度，不是认真刷数据。
        for _ in range(steps):
            t0 = time.perf_counter()  # 本步开始时间（高精度时钟）
            optimizer.zero_grad(set_to_none=True)  # 先清梯度，避免和上一步累加
            loss = model(**batch).loss  # 前向：batch 拆成 input_ids 等送进模型，取出 loss
            loss.backward()  # 反传：把梯度写进 params 里每个张量的 .grad
            optimizer.step()  # 按 SGD 用 .grad 真正改权重（LoRA 时只改补丁）
            last_loss = float(loss.detach().cpu())  # 断开计算图并拷到 CPU，变成普通 Python 数
            if device.type == "mps":
                # MPS 是异步的：不 synchronize，计时会偏短（GPU 还在算，Python 已经往下走了）
                sync = getattr(torch.mps, "synchronize", None)
                if callable(sync):
                    sync()  # 等到这块设备上的计算真正做完，再停表
            times.append(time.perf_counter() - t0)  # 本步耗时 = 现在 - 起步时刻
    except RuntimeError as e:
        # 内存不够时抓住，当成合法实验结果，不要让整段脚本崩掉。
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            oom = True
            err = str(e)
        else:
            raise  # 其它 RuntimeError（比如形状不对）不是本函数该吞的，原样抛出
    avg = sum(times) / len(times) if times else None  # 有成功步才算平均秒/步，否则 —
    return {
        "steps": steps,  # 计划跑几步（OOM 时实际可能更少，见 n_times 没有单独返回）
        "oom": oom,  # 是否 OOM
        "error": err,  # OOM 原文；成功时为空串
        "avg_step_s": avg,  # 平均每步多少秒
        "steps_per_s": (1.0 / avg) if avg else None,  # 每秒能走几步，avg 为空则没有
        "last_loss": last_loss,  # 最后一次成功 step 的 loss
        "rss_mb": rss_mb(),  # 进程内存高水位（粗），给 04～07 对照用
    }


def human_mb(path: Path) -> float | None:
    """普通文件大小（MB）。不存在或不是文件则返回 None，给 jsonl 用。"""
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def size_label(path: Path) -> str:
    """给人看的体积字符串：很小的 json 用 KB，权重大文件用 MB。"""
    if not path.exists():
        return "不存在"
    if path.is_dir():
        return "目录存在"
    n = path.stat().st_size
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def find_adapter_dirs() -> list[Path]:
    """在 checkpoints/ 和 models/ 里找出所有含 adapter_config.json 的目录。

    还没跑过 sft001 时列表为空，Day 2 脚本仍会把 adapter vs merge 的概念讲完。
    """
    found: list[Path] = []
    for root in (ROOT / "checkpoints", ROOT / "models"):
        if not root.exists():
            continue
        for cfg in root.rglob("adapter_config.json"):
            found.append(cfg.parent)
    return found


def lora_param_names(model, limit: int = 8) -> list[str]:
    """抽出名字里带 lora_ 的参数，默认只返回前几条给屏幕上看。

    典型长得像：...self_attn.q_proj.lora_A.default.weight
    用来确认旁路确实打在了 target_modules 上。
    """
    names = [n for n, _ in model.named_parameters() if "lora_" in n.lower()]
    return names[:limit]


def run_lora_tiny(
    *,
    script: str,
    tag: str,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    steps: int = 3,
    max_length: int = 64,
    batch_size: int = 1,
    extra: dict | None = None,
) -> dict:
    """一次完整的「小实验」：加载 135M → 注入指定 LoRA → 极少步 → 写入 jsonl。

    每次重新加载底座，避免 r=8 和 r=16 互相污染。
    返回的字典给 04/05/06 打表，08 复盘也会读 jsonl。
    extra 可并入记录（备用）。
    """
    targets = list(target_modules or ALL_MODULES)
    device = pick_device()
    model_dir = find_model_dir("135m")
    tokenizer, model = load_causal_lm(model_dir, device)
    total_full, train_full = count_params(model)
    config = make_lora_config(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
    )
    model = inject_lora(model, config)
    total_lora, train_lora = count_params(model)
    batch = make_batch(tokenizer, load_texts(batch_size), device, max_length=max_length)
    stats = timed_train_steps(model, batch, steps=steps)
    rec = {
        "script": script,
        "tag": tag,
        "device": str(device),
        "model": model_dir.name,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": targets,
        "n_target_modules": len(targets),
        "scale_alpha_over_r": (lora_alpha / r) if r else None,
        "full_params": total_full,
        "full_trainable": train_full,
        "lora_total_params": total_lora,
        "lora_trainable": train_lora,
        "trainable_pct": round(pct(train_lora, total_lora), 4),
        "batch_size": batch_size,
        "max_length": max_length,
        **stats,
    }
    if extra:
        rec.update(extra)
    append_run(rec)
    print(
        f"  [{tag}] 可训练 {train_lora:,} / 总 {total_lora:,}"
        f"  ({pct(train_lora, total_lora):.3f}%)"
        f"  秒/步={fmt(stats['avg_step_s'])}"
        f"  loss={fmt(stats['last_loss'])}"
        f"  OOM={fmt(stats['oom'])}"
    )
    if stats.get("error"):
        print(f"       错误: {stats['error'][:180]}")
    free_model(model)
    return rec


def fmt(v, digits=3) -> str:
    """把数字/布尔/空值打成对齐好的短字符串，表格里空缺显示为 —。"""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def append_run(record: dict) -> None:
    """把一条实验记录追加进 reports/week03_runs.jsonl，并打上时间戳。

    每行一个 JSON。08_week3_report.py 会读这个文件做周复盘。
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["ts"] = datetime.now().isoformat(timespec="seconds")
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

