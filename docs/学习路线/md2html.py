#!/usr/bin/env python3
"""将「学习路线」Markdown 按引用关系生成可 nginx 部署的静态 HTML 站点。

特性：
- 递归转换全部 .md → .html（目录镜像）
- .md / .md#锚点 / 目录/ → .html / .html#锚点 / README.html
- 保留材料中的 <a id="..."> 锚点，支持站内章节跳转
- 解析文档互相引用，生成侧边栏导航 + 文末「本页引用」
- 输出 nginx 示例配置

用法：
    python docs/学习路线/md2html.py
    python docs/学习路线/md2html.py --open
    python docs/学习路线/md2html.py --out /var/www/xuexi-route
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import webbrowser
from collections import defaultdict
from pathlib import Path


def ensure_markdown():
    try:
        import markdown  # noqa: F401
    except ImportError:
        import subprocess

        print("缺少依赖 markdown，正在安装…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])


ensure_markdown()
import markdown  # noqa: E402

ROOT = Path(__file__).resolve().parent
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.M)
TITLE_RE = re.compile(r"^#\s+(.+)$", re.M)


SITE_CSS = """
:root {
  --bg: #f4f1ea;
  --sidebar: #1c1917;
  --sidebar-text: #e7e5e4;
  --sidebar-muted: #a8a29e;
  --card: #fffdf8;
  --text: #1c1917;
  --muted: #57534e;
  --border: #e7e5e4;
  --link: #0f766e;
  --link-hover: #115e59;
  --code-bg: #f5f5f4;
  --accent: #c2410c;
  --ref-bg: #ecfdf5;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: "IBM Plex Sans", "PingFang SC", "Hiragino Sans GB",
    "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.75;
}
.layout {
  display: grid;
  grid-template-columns: 280px 1fr;
  min-height: 100vh;
}
.sidebar {
  background: var(--sidebar);
  color: var(--sidebar-text);
  padding: 20px 16px 40px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: auto;
}
.sidebar a { color: var(--sidebar-text); text-decoration: none; }
.sidebar a:hover { color: #fff; }
.sidebar .brand {
  font-weight: 700;
  font-size: 16px;
  letter-spacing: 0.02em;
  margin: 0 0 6px;
}
.sidebar .brand a { color: #fdba74; }
.sidebar .hint {
  color: var(--sidebar-muted);
  font-size: 12px;
  margin-bottom: 18px;
}
.sidebar h3 {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--sidebar-muted);
  margin: 18px 0 8px;
}
.sidebar ul {
  list-style: none;
  padding: 0;
  margin: 0 0 8px;
}
.sidebar li { margin: 0; }
.sidebar li a {
  display: block;
  padding: 7px 10px;
  border-radius: 8px;
  font-size: 13.5px;
  line-height: 1.4;
}
.sidebar li a.active {
  background: #292524;
  color: #fdba74;
}
.sidebar .sub {
  padding-left: 12px;
  margin-bottom: 10px;
}
.sidebar .sub a {
  font-size: 12.5px;
  color: var(--sidebar-muted);
  padding: 4px 10px;
}
.main {
  padding: 28px 28px 72px;
}
.crumb {
  font-size: 13px;
  color: var(--muted);
  margin-bottom: 14px;
}
.crumb a { color: var(--link); text-decoration: none; }
.crumb a:hover { text-decoration: underline; }
article {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 28px 34px 36px;
  box-shadow: 0 10px 30px rgba(28,25,23,.04);
  max-width: 920px;
}
article h1 { margin-top: 0; font-size: 1.85rem; }
article h2 {
  margin-top: 2em;
  padding-bottom: .35em;
  border-bottom: 1px solid var(--border);
}
article a { color: var(--link); }
article a:hover { color: var(--link-hover); }
code {
  font-family: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
  background: var(--code-bg);
  padding: .12em .38em;
  border-radius: 6px;
  font-size: .9em;
}
pre {
  background: #1c1917;
  color: #f5f5f4;
  border-radius: 10px;
  padding: 14px 16px;
  overflow: auto;
}
pre code { background: transparent; color: inherit; padding: 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
  font-size: .95em;
}
th, td {
  border: 1px solid var(--border);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}
th { background: var(--code-bg); }
blockquote {
  margin: 1em 0;
  padding: .3em 1em;
  border-left: 4px solid var(--accent);
  background: #fff7ed;
  color: var(--muted);
}
.refs {
  max-width: 920px;
  margin-top: 18px;
  background: var(--ref-bg);
  border: 1px solid #a7f3d0;
  border-radius: 12px;
  padding: 14px 18px;
}
.refs h3 { margin: 0 0 8px; font-size: 14px; color: #065f46; }
.refs ul { margin: 0; padding-left: 1.2em; }
.refs li { margin: 4px 0; font-size: 14px; }
.footer {
  max-width: 920px;
  margin-top: 14px;
  color: var(--muted);
  font-size: 12px;
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar {
    position: relative;
    height: auto;
    max-height: none;
  }
}
"""


NAV_TREE = [
    ("首页", "index.html"),
    ("学习大纲", "学习大纲.html"),
    ("学习计划", "学习计划.html"),
    ("README", "README.html"),
]

NAV_MATERIALS = [
    ("Encoder-Decoder 与问答流程", "材料/Encoder-Decoder与LLM问答流程.html"),
    ("微调学习笔记", "材料/微调学习笔记.html"),
    ("模型文件说明索引", "材料/模型文件说明/README.html"),
    ("01 目录与各文件总览", "材料/模型文件说明/01-目录与各文件总览.html"),
    ("02 tokenizer.json 结构", "材料/模型文件说明/02-tokenizer.json结构说明.html"),
]


def list_md_files(src_root: Path) -> list[Path]:
    return sorted(
        p
        for p in src_root.rglob("*.md")
        if "html" not in p.parts and p.name != "md2html.py"
    )


def page_title(md_text: str, fallback: str) -> str:
    m = TITLE_RE.search(md_text)
    return m.group(1).strip() if m else fallback


def rewrite_url(url: str) -> str:
    if re.match(r"^[a-z]+://", url, re.I) or url.startswith("mailto:"):
        return url
    if url.startswith("#"):
        return url

    path, frag = url, ""
    if "#" in url:
        path, frag = url.split("#", 1)
        frag = "#" + frag

    # 目录链接 → README.html
    if path.endswith("/"):
        path = path + "README.html"
    else:
        path = re.sub(r"\.md$", ".html", path)
        if path and not path.endswith(".html") and not Path(path).suffix:
            # 裸目录名
            path = path.rstrip("/") + "/README.html"

    return path + frag


def rewrite_md_links(md_text: str) -> str:
    def _sub(m: re.Match) -> str:
        text, url = m.group(1), m.group(2)
        return f"[{text}]({rewrite_url(url)})"

    return MD_LINK_RE.sub(_sub, md_text)


def extract_md_refs(md_text: str, src_file: Path, src_root: Path) -> list[Path]:
    """解析本页引用的其它 md 文件（相对路径解析到绝对 Path）。"""
    refs = []
    for _, url in MD_LINK_RE.findall(md_text):
        if re.match(r"^[a-z]+://", url, re.I) or url.startswith("#") or url.startswith("mailto:"):
            continue
        path = url.split("#", 1)[0]
        if not path:
            continue
        if path.endswith("/"):
            path = path + "README.md"
        elif not path.endswith(".md"):
            # 可能已是目录语义
            cand = (src_file.parent / path / "README.md").resolve()
            if cand.exists():
                refs.append(cand)
                continue
            continue
        target = (src_file.parent / path).resolve()
        try:
            target.relative_to(src_root)
        except ValueError:
            continue
        if target.exists() and target.suffix == ".md":
            refs.append(target)
    # unique keep order
    seen = set()
    out = []
    for p in refs:
        if p not in seen and p.suffix == ".md":
            seen.add(p)
            out.append(p)
    return out


def rel_href(from_html: Path, to_html: Path) -> str:
    return Path(to_html.as_posix()).as_posix()  # we'll compute properly below


def href_between(from_file: Path, to_file: Path) -> str:
    """from_file / to_file 都是相对于 out_root 的路径。"""
    return Path(to_file.as_posix()).as_posix()  # placeholder


def make_relative(from_rel: Path, to_rel: Path, fragment: str = "") -> str:
    back = Path("../" * (len(from_rel.parent.parts) if from_rel.parent != Path(".") else 0))
    # better use os.path.relpath logic
    import os

    start_dir = str(from_rel.parent) if from_rel.parent != Path(".") else "."
    target = str(to_rel)
    rel = os.path.relpath(target, start=start_dir)
    return rel.replace("\\", "/") + fragment


def build_sidebar(current_rel: Path, out_root_names: set[str]) -> str:
    def link(label: str, target: str) -> str:
        target_path = target.split("#", 1)[0]
        href = make_relative(current_rel, Path(target_path))
        active = " active" if current_rel.as_posix() == target_path else ""
        return f'<li><a class="{active.strip()}" href="{html.escape(href)}">{html.escape(label)}</a></li>'

    entrances = "\n".join(link(a, b) for a, b in NAV_TREE)
    materials = "\n".join(link(a, b) for a, b in NAV_MATERIALS)
    return f"""
<aside class="sidebar">
  <div class="brand"><a href="{html.escape(make_relative(current_rel, Path('index.html')))}">SmolLM 学习路线</a></div>
  <div class="hint">Javartisan</div>
  <h3>入口</h3>
  <ul>{entrances}</ul>
  <h3>材料</h3>
  <ul>{materials}</ul>
</aside>
"""


def render_md(md_text: str) -> str:
    return markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "nl2br", "attr_list"],
        output_format="html5",
    )


def convert_one(
    src: Path,
    src_root: Path,
    out_root: Path,
    graph: dict[Path, list[Path]],
    titles: dict[Path, str],
) -> Path:
    raw = src.read_text(encoding="utf-8")
    rewritten = rewrite_md_links(raw)
    body = render_md(rewritten)

    rel_md = src.relative_to(src_root)
    rel_html = rel_md.with_suffix(".html")
    dst = out_root / rel_html
    title = titles.get(src, src.stem)

    # breadcrumb
    crumb_parts = []
    home_href = make_relative(rel_html, Path("index.html"))
    crumb_parts.append(f'<a href="{html.escape(home_href)}">首页</a>')
    crumb_parts.append(html.escape(rel_html.as_posix()))

    # outgoing refs
    refs_html = ""
    outs = graph.get(src, [])
    if outs:
        lis = []
        for target in outs:
            t_rel = target.relative_to(src_root).with_suffix(".html")
            href = make_relative(rel_html, t_rel)
            label = titles.get(target, t_rel.as_posix())
            lis.append(
                f'<li><a href="{html.escape(href)}">{html.escape(label)}</a>'
                f' <code>{html.escape(t_rel.as_posix())}</code></li>'
            )
        refs_html = (
            '<section class="refs"><h3>本页引用的文档</h3><ul>'
            + "".join(lis)
            + "</ul></section>"
        )

    sidebar = build_sidebar(rel_html, set())

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)} · 学习路线</title>
  <style>{SITE_CSS}</style>
</head>
<body>
  <div class="layout">
    {sidebar}
    <div class="main">
      <div class="crumb">{" / ".join(crumb_parts)}</div>
      <article>
{body}
      </article>
      {refs_html}
      <div class="footer">由 md2html.py 生成 · 支持 nginx 静态部署 · 站内链接已转换为 HTML</div>
    </div>
  </div>
</body>
</html>
"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(page, encoding="utf-8")
    return dst


def build_home(src_root: Path, out_root: Path, graph: dict[Path, list[Path]], titles: dict[Path, str], md_files: list[Path]) -> Path:
    # recommended path cards
    cards = [
        ("1. 学习大纲", "学习大纲.html", "5 周目标与知识链"),
        ("2. 学习计划", "学习计划.html", "每日任务、验收、章节跳转入口"),
        ("3. 疑虑精读", "材料/Encoder-Decoder与LLM问答流程.html", "Encoder/Decoder/向量化"),
        ("4. 微调笔记", "材料/微调学习笔记.html", "SFT / LoRA / DPO"),
        ("5. 模型文件", "材料/模型文件说明/README.html", "config / tokenizer / 权重"),
    ]

    card_html = []
    for title, href, desc in cards:
        card_html.append(
            f'<li><a href="{html.escape(href)}"><strong>{html.escape(title)}</strong></a>'
            f' — {html.escape(desc)}</li>'
        )

    # reference edges
    edges = []
    for src, targets in graph.items():
        s = src.relative_to(src_root).with_suffix(".html").as_posix()
        for t in targets:
            tt = t.relative_to(src_root).with_suffix(".html").as_posix()
            edges.append(f"<li><code>{html.escape(s)}</code> → <code>{html.escape(tt)}</code></li>")

    all_pages = []
    for f in md_files:
        rel = f.relative_to(src_root).with_suffix(".html").as_posix()
        all_pages.append(
            f'<li><a href="{html.escape(rel)}">{html.escape(titles.get(f, rel))}</a> '
            f'<code>{html.escape(rel)}</code></li>'
        )

    rel_html = Path("index.html")
    sidebar = build_sidebar(rel_html, set())
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>学习路线站点首页</title>
  <style>{SITE_CSS}</style>
</head>
<body>
  <div class="layout">
    {sidebar}
    <div class="main">
      <div class="crumb">首页</div>
      <article>
        <h1>SmolLM 学习路线 · 静态站点</h1>
        <p>本站点由学习路线目录下的 Markdown 按<strong>引用关系</strong>生成，可直接用 nginx 托管，支持侧边栏导航与章节锚点跳转。</p>
        <h2>推荐阅读路径</h2>
        <ol>
          {''.join(card_html)}
        </ol>
        <h2>文档引用关系</h2>
        <p>以下为 Markdown 中显式链接解析出的引用边（A → B 表示 A 引用了 B）：</p>
        <ul>
          {''.join(edges) if edges else '<li>暂无解析到引用边</li>'}
        </ul>
        <h2>全部页面</h2>
        <ul>
          {''.join(all_pages)}
        </ul>
      </article>
      <div class="footer">部署时将本目录（html/）作为 nginx root 即可</div>
    </div>
  </div>
</body>
</html>
"""
    dst = out_root / "index.html"
    dst.write_text(page, encoding="utf-8")
    return dst


def write_nginx_example(out_root: Path) -> Path:
    conf = """# nginx 部署示例：把站点根指到本 html 目录
server {
    listen 8080;
    server_name localhost;

    # 改成你的实际路径，例如 /var/www/smollm-xuexi
    root REPLACE_WITH_ABS_PATH_TO_html;
    index index.html;

    location / {
        try_files $uri $uri/ $uri.html =404;
    }

    # 可选：关闭目录列表
    autoindex off;

    # 文本资源缓存（可按需调整）
    location ~* \\.(css|js|png|jpg|jpeg|gif|svg|ico)$ {
        expires 7d;
        add_header Cache-Control "public";
    }
}
"""
    path = out_root / "nginx.example.conf"
    path.write_text(
        conf.replace("REPLACE_WITH_ABS_PATH_TO_html", str(out_root.resolve())),
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser(description="学习路线 MD → 可部署 HTML 站点")
    parser.add_argument("--src", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    src_root = args.src.resolve()
    out_root = (args.out or (src_root / "html")).resolve()

    md_files = list_md_files(src_root)
    if not md_files:
        print(f"未找到 Markdown：{src_root}")
        sys.exit(1)

    # titles + graph
    titles: dict[Path, str] = {}
    graph: dict[Path, list[Path]] = {}
    for src in md_files:
        text = src.read_text(encoding="utf-8")
        titles[src] = page_title(text, src.stem)
        refs = extract_md_refs(text, src, src_root)
        graph[src] = [r for r in refs if r in md_files and r != src]

    print(f"源目录: {src_root}")
    print(f"输出:   {out_root}")
    print(f"文档数: {len(md_files)}")
    print(f"引用边: {sum(len(v) for v in graph.values())}\n")

    # clean old html pages under out? keep simple overwrite
    for src in md_files:
        dst = convert_one(src, src_root, out_root, graph, titles)
        print(f"  ✓ {src.relative_to(src_root)} → {dst.relative_to(out_root)}")

    index = build_home(src_root, out_root, graph, titles, md_files)
    nginx = write_nginx_example(out_root)
    print(f"\n首页: {index}")
    print(f"Nginx 示例: {nginx}")

    if args.open:
        webbrowser.open(index.as_uri())


if __name__ == "__main__":
    main()
