# 学习路线

本目录是学习路径的**独立完整包**：大纲、计划，以及计划中引用的阅读材料都在这里。

## 目录结构

```text
docs/学习路线/
├── README.md                 # 本说明
├── 学习大纲.md               # 5 周目标 + 第 1 周材料×代码对照
├── 学习计划.md               # 可执行详细计划（每日命令）
├── md2html.py                # Markdown → html/ 静态站点
├── gen_week01_page.py        # 汇总 week01 源码 + 运行结果
├── html/                     # 生成物，nginx 站点根（gitignore）
└── 材料/
    ├── 第1周知识体系与答疑.md
    ├── 第1周代码学习.md      # 由 gen_week01_page.py 生成
    └── 第1周代码运行结果/    # 各脚本 stdout 缓存
```

## 建议阅读顺序

1. [学习大纲.md](./学习大纲.md)  
2. [学习计划.md](./学习计划.md)（按第 0 阶段 → 第 1～5 周执行）  
3. 计划里写的阅读路径均相对本目录，例如：`材料/Encoder-Decoder与LLM问答流程.md`

## 转为 HTML 站点（可 nginx 部署）

```bash
# 在仓库根目录执行
python docs/学习路线/md2html.py
python docs/学习路线/md2html.py --open
# 重新跑 learn/week01 并把输出写进「第 1 周代码学习」页
python docs/学习路线/md2html.py --run-week01
```

生成目录：`docs/学习路线/html/`

站点能力：
- 全部 Markdown 转 HTML，目录结构镜像
- `.md` / `.md#锚点` / 目录链接 → `.html` / `.html#锚点` / `README.html`
- 左侧导航（入口 + 材料）
- 文末「本页引用的文档」（按 Markdown 引用关系解析）
- 首页展示推荐路径与引用关系图
- 附带 `nginx.example.conf`

本地预览：

```bash
cd docs/学习路线/html
python -m http.server 8080
# 浏览器打开 http://127.0.0.1:8080/
```

nginx 部署：把 `html/` 设为 `root`，可参考生成的 `html/nginx.example.conf`。

## 说明

- 代码在 `learn/`（按周练习）与 `scripts/`（训练/评测），数据在 `data/`，模型在 `models/`。  
- 概念类 Markdown 材料已全部收拢到本目录 `材料/`，避免学习时跳到外部 docs。
