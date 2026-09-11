"""第 3 周共用工具：注入 LoRA、数可训练参数、跑极少步、记实验。

本周一句话：冻住大模型权重，只训练旁路小补丁（A、B 矩阵）。
正式训练仍用 scripts/sft001.py；这里把概念拆开看。
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
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def pick_device() -> torch.device:
    return torch.device("mps") if mps_available() else torch.device("cpu")


def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024.0


def find_model_dir(size: str = "135m") -> Path:
    cands = MODELS_1P7 if size.lower().startswith("1.7") or size.lower().startswith("1p7") else MODELS_135
    for path in cands:
        if (path / "config.json").exists():
            return path
    kind = "1.7B" if cands is MODELS_1P7 else "135M"
    extra = " 可试：python learn/week03/07_try_1p7b.py --download" if kind == "1.7B" else ""
    raise FileNotFoundError(f"本地未找到 {kind}。{extra}")


def load_causal_lm(model_dir: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    return tokenizer, model


def free_model(model) -> None:
    del model
    gc.collect()
    if mps_available() and hasattr(torch, "mps"):
        fn = getattr(torch.mps, "empty_cache", None)
        if callable(fn):
            fn()


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def pct(trainable: int, total: int) -> float:
    return 100.0 * trainable / total if total else 0.0


def make_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """默认值对齐 scripts/sft001.py 的 build_lora_config。"""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules or ALL_MODULES),
    )


def inject_lora(model, config: LoraConfig):
    """冻底座，挂上 LoRA 旁路。返回 PeftModel。"""
    return get_peft_model(model, config)


def load_texts(n: int) -> list[str]:
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
    """极少步：只为看 loss/速度/谁在更新。不是完整 SFT。"""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr)
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    (model(**batch).loss).backward()
    optimizer.zero_grad(set_to_none=True)

    times: list[float] = []
    last_loss = None
    oom = False
    err = ""
    try:
        for _ in range(steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if device.type == "mps":
                sync = getattr(torch.mps, "synchronize", None)
                if callable(sync):
                    sync()
            times.append(time.perf_counter() - t0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            oom = True
            err = str(e)
        else:
            raise
    avg = sum(times) / len(times) if times else None
    return {
        "steps": steps,
        "oom": oom,
        "error": err,
        "avg_step_s": avg,
        "steps_per_s": (1.0 / avg) if avg else None,
        "last_loss": last_loss,
        "rss_mb": rss_mb(),
    }


def human_mb(path: Path) -> float | None:
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def size_label(path: Path) -> str:
    """给人看的体积：很小的 json 用 KB，权重用 MB。"""
    if not path.exists():
        return "不存在"
    if path.is_dir():
        return "目录存在"
    n = path.stat().st_size
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def find_adapter_dirs() -> list[Path]:
    """仓库里已有的 LoRA 适配器目录（含 adapter_config.json）。没有也正常。"""
    found: list[Path] = []
    for root in (ROOT / "checkpoints", ROOT / "models"):
        if not root.exists():
            continue
        for cfg in root.rglob("adapter_config.json"):
            found.append(cfg.parent)
    return found


def lora_param_names(model, limit: int = 8) -> list[str]:
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
    """加载 135M → 注入 LoRA → 极少步 → 记 jsonl。每次重新加载，避免配置互相污染。"""
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
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def append_run(record: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["ts"] = datetime.now().isoformat(timespec="seconds")
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
