#!/usr/bin/env python3
"""把 learn/week01 源码 + 一次真实运行输出，写成「第 1 周代码学习」Markdown。

用法（仓库根目录）：
    python docs/学习路线/gen_week01_page.py
    python docs/学习路线/gen_week01_page.py --run
    python docs/学习路线/md2html.py --run-week01
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WEEK01 = REPO / "learn" / "week01"
RUN_DIR = HERE / "材料" / "第1周代码运行结果"
OUT_MD = HERE / "材料" / "第1周代码学习.md"

# (文件名, 是否执行, 超时秒)
SCRIPTS: list[tuple[str, bool, int]] = [
    ("_common.py", False, 0),
    ("01_load_tokenizer.py", True, 180),
    ("02_token_and_id.py", True, 180),
    ("03_encode_decode.py", True, 180),
    ("04_chinese_vs_english.py", True, 180),
    ("05_special_tokens.py", True, 180),
    ("06_read_dataset.py", True, 60),
    ("07_forward_logits.py", True, 600),
    ("08_embedding.py", True, 600),
    ("09_loss_backward.py", True, 600),
    ("10_sft_messages.py", True, 180),
    ("11_eval_metrics.py", True, 600),
    ("12_week1_chain.py", True, 600),
]


def python_bin() -> str:
    venv = REPO / "venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def extract_learn_what(src: str) -> str:
    m = re.search(r"学什么\n((?:    .*\n)+)", src)
    if not m:
        first = src.strip().splitlines()[0] if src.strip() else ""
        return first.strip(' "')
    lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    return " ".join(lines)


def slug(name: str) -> str:
    return "w01-" + name.replace(".py", "").replace("_", "-")


def run_one(filename: str, timeout: int) -> tuple[int, str]:
    rel = f"learn/week01/{filename}"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    print(f"  运行 {rel} …", flush=True)
    try:
        proc = subprocess.run(
            [python_bin(), rel],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        return 124, (out + f"\n[超时] {timeout}s\n").strip()
    text = proc.stdout or ""
    if proc.stderr:
        text = text + ("\n" if text and not text.endswith("\n") else "") + proc.stderr
    return proc.returncode, text


def save_run(filename: str, code: int, output: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"# cmd: python learn/week01/{filename}\n"
        f"# exit: {code}\n"
        f"# captured: {stamp}\n"
        f"# {'-' * 40}\n"
    )
    (RUN_DIR / f"{filename}.txt").write_text(header + (output or "") + "\n", encoding="utf-8")


def load_run(filename: str) -> tuple[str | None, str | None, str]:
    path = RUN_DIR / f"{filename}.txt"
    if not path.exists():
        return None, None, ""
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body_lines = []
    in_body = False
    for line in raw.splitlines():
        if not in_body and line.startswith("# cmd:"):
            meta["cmd"] = line[6:].strip()
        elif not in_body and line.startswith("# exit:"):
            meta["exit"] = line[7:].strip()
        elif not in_body and line.startswith("# captured:"):
            meta["captured"] = line[12:].strip()
        elif not in_body and line.startswith("# -"):
            in_body = True
        elif in_body:
            body_lines.append(line)
        else:
            body_lines.append(line)
            in_body = True
    body = "\n".join(body_lines).rstrip() + "\n"
    return meta.get("exit"), meta.get("captured"), body


def fence(lang: str, text: str) -> str:
    ticks = "````"
    if "````" in text:
        ticks = "`````"
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}\n"


def build_markdown() -> str:
    toc = []
    sections = []
    for filename, runnable, _timeout in SCRIPTS:
        path = WEEK01 / filename
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        what = extract_learn_what(src)
        aid = slug(filename)
        toc.append(f"- [`{filename}`](#{aid}) — {what}")

        parts = [
            f"## `{filename}` {{#{aid}}}\n",
            f"{what}\n",
            f"路径：`learn/week01/{filename}`\n",
        ]
        if runnable:
            parts.append(f"运行（仓库根目录）：\n\n{fence('bash', f'python learn/week01/{filename}')}\n")
        else:
            parts.append("这是共用工具，不单独运行；被 `01`～`12` 导入。\n")

        parts.append("### 源码\n")
        parts.append(fence("python", src))
        parts.append("### 运行结果\n")
        if not runnable:
            parts.append("（库文件，无标准输出。）\n")
        else:
            exit_code, captured, body = load_run(filename)
            if not body:
                parts.append(
                    "尚未捕获运行输出。在仓库根目录执行：\n\n"
                    "`python docs/学习路线/md2html.py --run-week01`\n"
                )
            else:
                meta = []
                if captured:
                    meta.append(f"捕获时间：{captured}")
                if exit_code is not None:
                    meta.append(f"退出码：{exit_code}")
                if meta:
                    parts.append("；".join(meta) + "\n")
                parts.append(fence("text", body))
        sections.append("\n".join(parts))

    intro = f"""# 第 1 周 代码学习

对照仓库 `learn/week01/`：每个脚本给出**完整源码**和一次在本机捕获的**运行输出**。

概念讲解仍看 [第1周知识体系与答疑.md](./第1周知识体系与答疑.md)（尤其 [第 7 节代码导读](./第1周知识体系与答疑.md#code-guide)）。

运行约定：仓库根目录，`source venv/bin/activate`。刷新输出：

```bash
python docs/学习路线/md2html.py --run-week01
```

## 目录

{chr(10).join(toc)}

---
"""
    return intro + "\n\n---\n\n".join(sections) + "\n"


def write_markdown() -> Path:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_markdown(), encoding="utf-8")
    return OUT_MD


def run_all() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for filename, runnable, timeout in SCRIPTS:
        if not runnable:
            continue
        if not (WEEK01 / filename).exists():
            print(f"  跳过缺失文件 {filename}")
            continue
        code, output = run_one(filename, timeout)
        save_run(filename, code, output)
        print(f"    exit={code}  bytes={len(output.encode('utf-8'))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成第 1 周代码学习 Markdown")
    parser.add_argument("--run", action="store_true", help="重新执行 week01 脚本并保存输出")
    args = parser.parse_args()
    if args.run:
        run_all()
    path = write_markdown()
    print(f"已写入 {path}")


if __name__ == "__main__":
    main()
