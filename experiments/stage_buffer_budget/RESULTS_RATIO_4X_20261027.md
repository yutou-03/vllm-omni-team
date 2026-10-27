# 2026-10-27：Thinker:Talker 4:1／4:2／4:3 实测

服务源码固定在 `9fb7ed5f5d7d89bc019d82b83e46d139b31d0174`；三组使用相同 SHA-256 的 280 条／30 秒多模态负载，并只改变 Talker 的独立 TP1 副本数及设备列表。Thinker 保持 GPU 0–3、TP4，Code2Wav 保持 GPU 5。三组均完整尝试 280 条、服务端排空、无运行错误，全部原始文件校验见 [运行目录](/home/zhongyu/research_reports/kv_resource_ratios_20261027/ratio_4x_280_in_30s/study_manifest.json)。

| 配比 | 成功／280 | 240 s 安全超时 | 完成请求／270 s | 接受音频内容秒／墙钟秒 | Talker 同时 KV 峰值 | Talker 被动抢占 |
|---|---:|---:|---:|---:|---:|---:|
| 4:1 | 179 | 101 | 0.663/s | 10.33 | 1.894 GB | 14 |
| 4:2 | 270 | 10 | 1.000/s | 19.31 | 3.788 GB | 17 |
| 4:3 | 277 | 3 | 1.026/s | 20.18 | 5.605 GB | 16 |

三个配比的各个 Talker 池都曾达到 100% 占用。增加 Talker GPU 改善了这次负载的完成量，但没有消除内存不足的被动抢占；额外硬件同时增加算力、执行名额和 KV 容量，尚不能分离各自贡献。用户尚未确定 SLO，因此这里报告的是实际完成和音频产出，**不报告 SLO goodput**。每种配比只有一次运行，不能将差值当作有置信区间的普遍结论。指标为原生 allocator 池块占用，而非请求持有的 `B_live`。

[完整报告与实测图](/home/zhongyu/research_reports/kv_resource_ratios_20261027/ratio_4x_280_in_30s/results.md)；[可执行复现说明](README_RATIO_4X.md)。
