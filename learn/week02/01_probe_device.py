"""Day 1：探测本机实际能用什么设备。

学什么
    训练前先问三件事：PyTorch 版本、MPS 能不能用、脚本会选哪块设备。
    Intel Mac 上官方轮子常停在较旧版本，MPS 可能不可用——这不是你操作错。

对应阅读
    docs/学习路线/学习计划.md 第 2 周 Day 1

运行
    python learn/week02/01_probe_device.py

----------------------------------------------------------------------
科普：CPU / MPS / CUDA 各是什么

    CPU     中央处理器。第 1 周用它把概念跑通，慢但稳。
    CUDA    NVIDIA 显卡上的计算接口。本仓库当前环境是 Mac，通常没有。
    MPS     Metal Performance Shaders：苹果 GPU 的 PyTorch 后端。
            Apple Silicon（M 系列）上较成熟；Intel Mac + 旧 PyTorch 常常 is_available()==False。

本周目标不是「必须用上 GPU」，而是：
    测得到 → 记下来 → 后面对比实验有依据。
    MPS 没有，就改做 CPU 上不同 batch 的对比，概念同样成立。
"""

from _common import append_run, device_report, mps_available, pick_device, rss_mb


def main():
    info = device_report()
    print("=" * 60)
    print("【1】PyTorch 与设备探测")
    print("=" * 60)
    print(f"  torch.__version__              = {info['torch']}")
    print(f"  python                         = {info['python']}")
    print(f"  platform                       = {info['platform']}")
    print(f"  torch.backends.mps.is_built()  = {info.get('mps_is_built', '（无此属性）')}")
    print(f"  torch.backends.mps.is_available() = {info['mps_is_available']}")
    print(f"  pick_device()                  = {info['picked']}")
    print(f"  当前进程 RSS 约                 = {rss_mb():.1f} MB（加载大模型前）")

    print()
    print("=" * 60)
    print("【2】怎么读这三个结果")
    print("=" * 60)
    print("  is_built()      : 这份 PyTorch 编译时有没有带 MPS 代码。")
    print("  is_available()  : 这台机器现在能不能真的用 MPS。")
    print("  pick_device()   : 本周后续脚本默认往哪送模型和张量。")

    print()
    if mps_available():
        print("结论：本机 MPS 可用。Day 2～3 可以做 CPU vs MPS 对比。")
        note = "MPS 可用，后续对比 CPU 与 MPS。"
    else:
        print("结论：本机 MPS 不可用（Intel Mac + 旧轮子上很常见）。")
        print("      第 2 周改为：同一模型、CPU 上对比不同 batch size。")
        print("      计划里的对比表 MPS 那一行填「不可用」即可，不算不及格。")
        note = "MPS 不可用，后续以 CPU 不同 batch 为主。"

    print()
    print("笔记作业（请手写三句话）：")
    print("  1. 我的 torch 版本是 ____")
    print("  2. MPS 是否 available：____")
    print("  3. 本周实验默认设备：____")

    append_run({"script": "01_probe_device", **info, "note": note})
    print()
    print("已写入 reports/week02_runs.jsonl （07 复盘会读）")


if __name__ == "__main__":
    main()
