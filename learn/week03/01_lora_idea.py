"""Day 1：LoRA 在算什么（不加载大模型）。

学什么
    全量微调要改整张权重矩阵 W。
    LoRA 冻住 W，只在旁边加两张很小的矩阵 A、B，用 BA 去近似「该怎么改」。

对应阅读
    docs/学习路线/lora_animation.html（冻底座 + 旁路动画；站点 Day 1 互链）
    docs/学习路线/材料/微调学习笔记.md 第 6 节
    scripts/sft001.py 的 build_lora_config（先看注释，下一脚本再对照真模型）

运行
    python learn/week03/01_lora_idea.py

----------------------------------------------------------------------
科普：冻底座 + LoRA 旁路

    原来一层线性是：

        y = W x

    全量微调：W 的每一个数字都可以被梯度改掉。
    若 W 是 576×576，这一层就要训 331776 个参数。

    LoRA 改成：

        y = W x + 缩放 × B (A x)
              ↑           ↑
           冻住不更新    只训 A 和 B

        A 形状 [r, in]     很扁
        B 形状 [out, r]    很瘦
        缩放 ≈ lora_alpha / r

    图（请自己再画一遍）：

        x ──► [冻住的 W] ──► + ──► y
              ↘ [A] → [B] ↗
                 （可训练旁路）

    为什么能省参数？
        全量：out × in
        LoRA：r × (in + out)
        r 远小于 in/out 时，乘积会小一个数量级。

    三个名字先记住：
        r              秩。越大，补丁越能「记住细节」，也更占内存、更容易过拟合。
        lora_alpha     和 r 一起决定旁路放大多少。本仓库 sft001 用 alpha=32、r=16，缩放=2。
        target_modules 给哪些线性层挂旁路。常见是注意力的 q/k/v/o，再加 MLP 的 gate/up/down。
"""

from _common import ALL_MODULES, append_run


def lora_params(in_f: int, out_f: int, r: int) -> int:
    """数这一层挂 LoRA 时要训多少参数（不是在训练网络）。

    线性层权重 W 形状约 [out_f, in_f]。LoRA 冻住 W，另学两张小矩阵：
        A：形状约 [r, in_f]  → 参数个数 r * in_f
        B：形状约 [out_f, r] → 参数个数 r * out_f
    合计：r * in_f + r * out_f = r * (in_f + out_f)

    小例子（手算一遍就懂）：
        设 in_f=4, out_f=3, r=2

        全量 W（冻住，不训）：
            W 是 3×4，共 12 个数，例如
                [[w11, w12, w13, w14],
                 [w21, w22, w23, w24],
                 [w31, w32, w33, w34]]

        LoRA 只训 A、B：
            A 是 2×4（r × in）→ 2*4 = 8 个数
                [[a11, a12, a13, a14],
                 [a21, a22, a23, a24]]
            B 是 3×2（out × r）→ 3*2 = 6 个数
                [[b11, b12],
                 [b21, b22],
                 [b31, b32]]

            前向旁路：x(长度4) → A → (长度2) → B → (长度3)
            可训练参数 = 8 + 6 = 14
            也等于 r*(in_f+out_f) = 2*(4+3) = 14

        对照：全量要训 3*4=12；本例 r 相对很大，LoRA 不一定更省。
        真实模型里 r≪in/out（如 576 维、r=8）时才会明显变少：
            r=8, in=out=576 → 8*(576+576)=9216 ≪ 576*576=331776。
    """
    return r * (in_f + out_f)


def full_params(in_f: int, out_f: int) -> int:
    """数这一层全量微调要训多少参数。

    整张 W（in_f × out_f）每个元素都可被梯度更新，
    参数个数就是矩阵元素个数：in_f * out_f。

    例：in_f=out_f=576 → 576*576=331776。
    常与 lora_params 对比压缩比：lora_params / full_params。
    """
    return in_f * out_f


def main():
    print("=" * 60)
    print("【1】一张线性层：全量 vs LoRA 参数量")
    print("=" * 60)
    in_f = out_f = 576  # 接近 SmolLM2-135M 隐藏维度，方便建立数量级
    print(f"  假设一层线性：in = out = {in_f}")
    print(f"  {'r':<8}{'全量参数':<14}{'LoRA 参数':<14}{'LoRA/全量'}")
    for r in (4, 8, 16, 32, 64):
        full = full_params(in_f, out_f)
        lora = lora_params(in_f, out_f, r)
        print(f"  {r:<8}{full:<14,}{lora:<14,}{100.0 * lora / full:.2f}%")

    print()
    print("=" * 60)
    print("【2】缩放 lora_alpha / r")
    print("=" * 60)
    print("  旁路不是原样加回去，而要乘一个系数，避免一开始就把底座打乱。")
    print("  本仓库 scripts/sft001.py：r=16, lora_alpha=32  →  缩放 = 32/16 = 2")
    print()
    print(f"  {'r':<8}{'alpha':<10}{'缩放 alpha/r'}")
    for r, alpha in ((8, 32), (16, 32), (16, 16), (8, 16)):
        print(f"  {r:<8}{alpha:<10}{alpha / r:.2f}")
    print()
    print("  注意：只改 r、不改 alpha 时，缩放也会变。")
    print("  Day 3 对比 r=8 vs 16 时，我们故意固定 alpha=32（和 sft001 一致），")
    print("  所以 r=8 时缩放变成 4。这是「一次只改一个旋钮」的代价，心里有数即可。")

    print()
    print("=" * 60)
    print("【3】target_modules 打在哪")
    print("=" * 60)
    print("  sft001 默认挂在这些投影上：")
    print(f"    {ALL_MODULES}")
    print("  注意力：q/k/v/o_proj    MLP：gate/up/down_proj")
    print("  挂得越多，可训练参数越多、表达力越强，也更容易过拟合小数据。")

    print()
    print("=" * 60)
    print("【4】全量微调 vs LoRA（一句话对照）")
    print("=" * 60)
    print("  全量：每个权重都能改。模型大、数据少时又慢又容易背答案。")
    print("  LoRA：底座当「已经会说话的引擎」，只训一小块补丁去贴近你的任务。")
    print("  小数据（本仓库 Java 问答只有很少条）更适合 LoRA，而不是把 135M 全拧一遍。")

    print()
    print("笔记作业（请手写）：")
    print("  1. 用自己的话画出「冻 W + 旁路 BA」。")
    print("  2. r 变大，可训练参数怎么变？")
    print("  3. 为什么小数据更愿意用 LoRA，而不是全量？")

    append_run(
        {
            "script": "01_lora_idea",
            "demo_in": in_f,
            "demo_r16_lora": lora_params(in_f, out_f, 16),
            "demo_r16_full": full_params(in_f, out_f),
        }
    )
    print()
    print("已写入 reports/week03_runs.jsonl（08 复盘会读）")
    print("下一步：python learn/week03/02_inject_and_count.py")


if __name__ == "__main__":
    main()
