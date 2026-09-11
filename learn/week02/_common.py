"""第 2 周共用工具：设备探测、加载模型、计时、记内存。

和第 1 周的差别：
    week01/_common.py 把模型固定放到 CPU，先把数据流跑明白。
    本周要在 CPU / MPS 之间切换，并观察 batch size、更大模型。

你暂时不必把这个文件背下来，知道下面几个函数即可：
    pick_device()       本机实际能用 cpu 还是 mps
    load_causal_lm()    按目录加载，并 .to(device)
    make_batch()        从 train.jsonl 组一个小 batch
    timed_train_steps() 跑几步 Forward → Loss → Backward → step，返回耗时/内存
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

# ---------------------------------------------------------------------------
# Intel Mac + PyTorch 2.2.2：新版 transformers 会调 mps.is_macos_or_newer，
# 2.2.2 没有这个函数，不补会在加载时直接崩。
# ---------------------------------------------------------------------------
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
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_FILE = ROOT / "data" / "train.jsonl"
REPORTS_DIR = ROOT / "reports"
RUNS_FILE = REPORTS_DIR / "week02_runs.jsonl"

HF_135 = "HuggingFaceTB/SmolLM2-135M-Instruct"
HF_360 = "HuggingFaceTB/SmolLM2-360M-Instruct"

MODELS_135 = [
    ROOT / "models" / "SmolLM2-135M-Instruct",
    ROOT / "models" / "SmolLM2-135M",
    ROOT / "models" / "smollm2-135m-java-sft",
    ROOT / "models" / "smollm2-135m-java",
]
MODELS_360 = [
    ROOT / "models" / "SmolLM2-360M-Instruct",
    ROOT / "models" / "SmolLM2-360M",
]


def patch_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def find_model_dir(size: str = "135m") -> Path:
    """size: '135m' 或 '360m'。目录里要有 config.json。"""
    cands = MODELS_360 if size.lower().startswith("360") else MODELS_135
    for path in cands:
        if (path / "config.json").exists():
            return path
    kind = "360M" if size.lower().startswith("360") else "135M"
    extra = ""
    if kind == "360M":
        extra = " 可运行：python learn/week02/06_try_360m.py --download"
    raise FileNotFoundError(f"本地未找到 {kind} 模型。{extra}")


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def pick_device() -> torch.device:
    """本机优先 MPS，否则 CPU。第 2 周的「能用什么」以这个为准。"""
    return torch.device("mps") if mps_available() else torch.device("cpu")


def device_report() -> dict:
    info = {
        "torch": torch.__version__,
        "mps_is_available": mps_available(),
        "picked": str(pick_device()),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    if hasattr(torch.backends, "mps"):
        info["mps_is_built"] = torch.backends.mps.is_built()
    return info


def rss_mb() -> float:
    """进程内存高水位（粗）。macOS 单位是字节，Linux 一般是 KB。"""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024.0


def device_alloc_mb(device: torch.device) -> float | None:
    if device.type == "mps" and hasattr(torch, "mps"):
        fn = getattr(torch.mps, "current_allocated_memory", None)
        if callable(fn):
            return fn() / (1024 * 1024)
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return None


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


def load_texts(n: int) -> list[str]:
    """把 prompt+completion 拼成短文本，只为组 batch，不是认真微调。"""
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
    print(f"enc = {enc}")
    batch = {k: v.to(device) for k, v in enc.items()}
    batch["labels"] = batch["input_ids"].clone()
    return batch


def timed_train_steps(
        model,
        batch: dict,
        *,
        steps: int = 3,
        accum: int = 1,
        lr: float = 1e-4,
        warmup: bool = True,
) -> dict:
    """跑极少步训练，用来比速度/内存。不是完整 SFT。

    accum>1 时：连续 accum 次 backward 再 step，模拟更大有效 batch。
    本练习重复使用同一个 batch，只为把「累积」看清楚。
    """
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr)
    device = next(model.parameters()).device

    if warmup:
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss / max(accum, 1)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    rss_before = rss_mb()
    dev_before = device_alloc_mb(device)
    times: list[float] = []
    last_loss = None
    oom = False
    err = ""

    try:
        for _ in range(steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _micro in range(max(accum, 1)):
                loss = model(**batch).loss / max(accum, 1)
                loss.backward()
                last_loss = float(loss.detach().cpu()) * max(accum, 1)
            optimizer.step()
            if device.type == "mps":
                sync = getattr(torch.mps, "synchronize", None)
                if callable(sync):
                    sync()
            times.append(time.perf_counter() - t0)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "oom" in msg:
            oom = True
            err = str(e)
        else:
            raise

    rss_after = rss_mb()
    dev_after = device_alloc_mb(device)
    avg = sum(times) / len(times) if times else None
    return {
        "steps": steps,
        "accum": accum,
        "oom": oom,
        "error": err,
        "avg_step_s": avg,
        "steps_per_s": (1.0 / avg) if avg else None,
        "last_loss": last_loss,
        "rss_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "device_alloc_mb": dev_after,
        "device_alloc_delta_mb": (
            None if dev_before is None or dev_after is None else dev_after - dev_before
        ),
        "n_times": len(times),
    }


def append_run(record: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record["ts"] = datetime.now().isoformat(timespec="seconds")
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def fmt(v, digits=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)
