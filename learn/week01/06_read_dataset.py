"""Day 2：看一条训练样本长什么样。

学什么
    本项目 SFT 数据是 jsonl：每一行一个 JSON。
    prompt     = 用户问题
    completion = 标准答案（模型要学会的回复）
    训练集拿来更新参数；测试集只用来检查，绝不能混进训练。

对应阅读
    data/train.jsonl、data/test.jsonl
    docs/学习路线/材料/微调学习笔记.md 第 2 节

运行
    python learn/week01/06_read_dataset.py
"""

import json

from _common import TEST_FILE, TRAIN_FILE


def load_jsonl(path):
    """按行读取 jsonl，返回 list[dict]。"""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def preview(name: str, path, rows, n=3):
    print("=" * 60)
    print(f"{name}: {path}")
    print("=" * 60)
    print(f"条数 = {len(rows)}")
    print(f"字段 = {list(rows[0].keys()) if rows else '(空文件)'}")
    print()
    for i, row in enumerate(rows[:n], start=1):
        print(f"  样本 {i}")
        print(f"    prompt     : {row.get('prompt')}")
        print(f"    completion : {row.get('completion')}")
        print()


def main():
    train_rows = load_jsonl(TRAIN_FILE)
    test_rows = load_jsonl(TEST_FILE)

    preview("训练集 TRAIN", TRAIN_FILE, train_rows)
    preview("测试集 TEST", TEST_FILE, test_rows)

    print("=" * 60)
    print("为什么测试集不能拿去训练？")
    print("=" * 60)
    print("  训练 = 用标准答案改参数，模型会越来越像训练集。")
    print("  如果测试题也拿去训练，考试时等于「提前看过答案」，分数虚高。")
    print("  所以 test.jsonl 只在评测脚本（eval_compare.py）里使用。")

    print()
    print("=" * 60)
    print("作业（请你自己动手，本脚本不会改数据文件）")
    print("=" * 60)
    print("  在 data/train.jsonl 末尾新增 2 条短 Java 问答，格式完全一致：")
    print('  {"prompt":"你的问题","completion":"一两句标准答案"}')
    print("  注意：不要写进 test.jsonl。")

    print()
    print("小结论：")
    print("  - prompt 是问题，completion 是要模仿的答案。")
    print("  - 训练/测试必须分开，否则评测不可信。")


if __name__ == "__main__":
    main()
