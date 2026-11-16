# 中文 KWS 示例音频

音频来自 [OpenSLR 87 MobvoiHotwords](https://www.openslr.org/87/)，许可为 Apache 2.0。
选取官方测试划分中的两个唤醒词和一段背景语音。

| 文件 | 内容 | keyword_id |
|---|---|---:|
| `01_hi_xiaowen.wav` | Hi 小问 | 0 |
| `02_nihao_wenwen.wav` | 你好问问 | 1 |
| `03_background.wav` | 背景语音 | -1 |

原始文件名、SHA-256、说话人和官方划分记录在 [manifest.json](manifest.json)。
构建时将短音频补零到 31920 个样本，保持 16 kHz、单声道 PCM16。
