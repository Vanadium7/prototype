# AIGC 图像溯源系统说明
本项目为东南大学网络空间安全前沿课程论文实验代码，包含论文中 AIGC 图像内容溯源系统实验的代码、输入图像、日志和实验结果。

## 运行环境

本实验使用的环境如下：

| 项目 | 版本 |
|---|---|
| Python | 3.13.9 |
| NumPy | 2.3.5 |
| Pillow | 12.0.0 |
| SciPy | 1.16.3 |

## 文件结构

| 路径 | 说明 |
|---|---|
| `aigc_trace_demo.py` | 主实验脚本 |
| `inputs/` | 20 张 512 x 512 AIGC 测试图像 |
| `logs/` | 每张图像对应的 JSON 溯源记录 |
| `outputs/` | 含水印图像和攻击后图像 |
| `results/summary_results.csv` | 按攻击类型汇总的平均实验结果 |
| `results/detailed_results.csv` | 每张图像的详细实验结果 |
| `results/summary_results.md` | Markdown 格式实验结果汇总 |
| `results/baseline_comparison.csv` | 本文方案与 DCT 单次嵌入 baseline 的对比结果 |
| `results/baseline_comparison.md` | Markdown 格式 baseline 对比结果 |
| `results/ablation_results.md` | 不同证据组合的消融对比结果 |
| `results/verification_trace_samples.json` | 示例验证流程记录 |
| `results/visual_comparison.png` | 实验图像可视化对比 |
| `results/generated_aigc_contact_sheet.png` | 补充生成的 AIGC 图像拼图源文件 |

## 关键参数

| 项目 | 设置 |
|---|---|
| 水印载荷 | 64 bit `WatermarkID` |
| 本文方案重复嵌入因子 | 7 |
| Baseline 重复嵌入因子 | 1 |
| DCT 分块大小 | 8 x 8 |
| DCT 系数位置 | `(3, 4)` 与 `(4, 3)` |
| 嵌入阈值 | 42 |
| pHash 大小 | 8 x 8 |
| 高相似 pHash 阈值 | 8 |
| 中等相似 pHash 阈值 | 12 |
| 签名方式 | HMAC-SHA256 |

## 攻击设置

| 攻击类型 | 参数 |
|---|---|
| JPEG 压缩 | 质量因子 90、70、50 |
| 高斯噪声 | sigma = 0.01 |
| 缩放攻击 | 缩小至 50% 后恢复尺寸 |
| 裁剪攻击 | 中心裁剪 10% 后恢复尺寸 |
| 局部遮挡篡改 | 覆盖两个局部矩形区域 |
| 亮度调整 | +20% |
| 模糊攻击 | 高斯模糊半径 1.2 |
| 日志签名篡改 | 保持图像不变，篡改日志签名 |

## 复现实验

在 `prototype` 目录下运行：

```powershell
python .\aigc_trace_demo.py
```

脚本会重新生成含水印图像、攻击后图像、JSON 日志和实验结果文件，输出目录包括 `outputs/`、`logs/` 和 `results/`。

如果 `inputs/` 目录为空或图像数量不足，脚本可生成确定性的合成测试图像用于补充。
