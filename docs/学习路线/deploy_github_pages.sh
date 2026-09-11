#!/usr/bin/env bash
# 把学习路线 html/ 同步到 GitHub Pages 仓库的 llm/ 目录并 push。
#
# 用法（建议在本仓库根目录）：
#   bash docs/学习路线/deploy_github_pages.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HTML_DIR="$SCRIPT_DIR/html"
PAGES_ROOT="/Users/tengdelong/javartisan.github.io"
DEST="$PAGES_ROOT/llm"
ORIG_PWD="$(pwd)"

cleanup() {
  cd "$ORIG_PWD"
}
trap cleanup EXIT

if [[ ! -d "$PAGES_ROOT/.git" ]]; then
  echo "找不到 GitHub Pages 仓库：$PAGES_ROOT"
  exit 1
fi

if [[ -x "$LAB_ROOT/venv/bin/python" ]]; then
  PYTHON="$LAB_ROOT/venv/bin/python"
else
  PYTHON="python3"
fi

echo "==> 生成 html/"
cd "$LAB_ROOT"
"$PYTHON" "$SCRIPT_DIR/md2html.py"

if [[ ! -d "$HTML_DIR" ]] || [[ -z "$(ls -A "$HTML_DIR" 2>/dev/null || true)" ]]; then
  echo "html 目录为空：$HTML_DIR"
  exit 1
fi

echo "==> 清空 $DEST"
mkdir -p "$DEST"
# 只清空 llm/ 内容，不动 github.io 仓库其它文件
find "$DEST" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

echo "==> 复制 html/ → llm/"
rsync -a --delete \
  --exclude '.DS_Store' \
  --exclude '**/.DS_Store' \
  "$HTML_DIR"/ "$DEST"/

echo "==> 在 $PAGES_ROOT 提交并推送"
cd "$PAGES_ROOT"
git add llm
if git diff --cached --quiet; then
  echo "llm/ 没有变更，跳过 commit / push"
else
  git commit -m "$(cat <<'EOF'
docs: 更新 llm 学习路线静态站点

EOF
)"
  git push
fi

echo "==> 回到 $ORIG_PWD"
