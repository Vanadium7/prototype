# AIGC Trace Prototype Results

- Images: 20
- Watermark bits: 64
- Repetition factor: 7
- DCT threshold: 42.0

## Summary

| 攻击类型 | PSNR | SSIM | Bit Accuracy | BER | NC | pHash距离 | 日志匹配率 | 主要风险判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 无攻击 | 42.368 | 0.9993 | 1.0 | 0.0 | 1.0 | 0.2 | 1.0 | 高可信来源一致 |
| JPEG Q=90 | 36.158 | 0.9983 | 1.0 | 0.0 | 1.0 | 0.1 | 1.0 | 高可信来源一致 |
| JPEG Q=70 | 33.164 | 0.9952 | 0.9969 | 0.0031 | 0.9938 | 0.1 | 0.9 | 高可信来源一致 |
| JPEG Q=50 | 32.221 | 0.9933 | 0.6664 | 0.3336 | 0.3328 | 0.0 | 0.0 | 内容相似，但水印可能被破坏或移除 |
| 高斯噪声 sigma=0.01 | 37.57 | 0.9984 | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 | 高可信来源一致 |
| 缩放 0.5x | 31.231 | 0.9813 | 0.8641 | 0.1359 | 0.7281 | 0.1 | 0.0 | 内容相似，但水印可能被破坏或移除 |
| 裁剪 10% | 14.016 | 0.4384 | 0.4977 | 0.5023 | -0.0047 | 18.3 | 0.0 | 内容变化较大，无法可信溯源 |
| 局部遮挡篡改 | 14.872 | 0.6425 | 1.0 | 0.0 | 1.0 | 17.0 | 1.0 | 来源记录存在，但内容疑似大幅篡改 |
| 亮度 +20% | 24.382 | 0.974 | 1.0 | 0.0 | 1.0 | 0.9 | 1.0 | 高可信来源一致 |
| 模糊 r=1.2 | 29.417 | 0.9729 | 1.0 | 0.0 | 1.0 | 0.2 | 1.0 | 高可信来源一致 |
| 日志签名篡改 | 42.368 | 0.9993 | 1.0 | 0.0 | 1.0 | 0.2 | 1.0 | 疑似水印伪造或来源冒用 |

## Notes

- `log_match_rate` is strict exact matching between extracted WatermarkID and the platform log.
- Partial watermark recovery is still reported through Bit Accuracy, BER, and NC.
- Risk labels combine watermark recovery, signature/log matching, and pHash distance.

## Detailed CSV

- `detailed_results.csv` contains per-image rows.
- `summary_results.csv` contains attack-level averages.