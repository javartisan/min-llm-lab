# 按周学习脚本

这里只放「这一周要动手跑」的小脚本，和 `scripts/` 里的正式训练入口分开。

| 目录 | 对应计划 | 主题 |
|---|---|---|
| [week01/](./week01/) | 第 1 周 | Tokenizer、Dataset、Forward、Loss、Backward、SFT 格式、评测直觉 |
| [week02/](./week02/) | 第 2 周 | 探测设备、CPU vs MPS、batch、梯度累积、360M；配合 [Auto 工厂文档](../docs/学习路线/材料/第2周-BaseAutoModelClass与Auto家族.md) |
| [week03/](./week03/) | 第 3 周 | LoRA：冻底座、只训补丁；对照全量见 `09_full_sft_135m.py` |

一律在**仓库根目录**运行，例如：

```bash
python learn/week01/01_load_tokenizer.py
python learn/week02/01_probe_device.py
python learn/week03/01_lora_idea.py
```
