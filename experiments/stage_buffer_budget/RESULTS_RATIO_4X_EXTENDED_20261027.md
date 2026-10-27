# 2026-10-27：280 条／30 秒同轨迹的 4:1、4:2、4:3 更正实验

旧客户端的 240 秒 watchdog 按**每条请求计划到达时刻**主动关闭未完成的 SSE 连接。它不是全局停止接收请求，也不是纯统计截尾。先前 4:1 的 179／280 因此只表示该限制内完成的数量。放宽为 900 秒安全上限后，同一轨迹的三组均自然完成 280／280，无超时、无运行错误，服务端排空检查通过；900 秒不是 SLO。

| Thinker:Talker | 完成／280 | 实际排空跨度 | 完成请求／墙钟秒 | 客户端接受音频内容秒／墙钟秒 | TTFP P50/P95 | Talker 被动抢占 |
|---|---:|---:|---:|---:|---:|---:|
| 4:1 | 280 | 452.13 s | 0.619 | 12.142 | 156.85 / 291.71 s | 24 |
| 4:2 | 280 | 302.44 s | 0.926 | 18.156 | 90.43 / 181.87 s | 12 |
| 4:3 | 280 | 264.00 s | 1.061 | 20.842 | 73.89 / 158.40 s | 13 |

服务源码与扩展客户端来自固定提交 `2a90b2d5ebe6e8dc10008b8026d7a18955d672d7`；负载 SHA-256 为 `5c3b04742649a4bd694a46b7850fab6e8fd6e9d40cb50c4a1e6edca867197693`。4:2 使用物理 GPU 4、7 作为两个 Talker 副本，因为当时 GPU 6 被其他任务占用；YAML 里的逻辑设备号 `4,6` 在可见列表 `0,1,2,3,4,5,7` 中映射到物理 `4,7`。物理 PID 映射见运行清单。

排空产出率是有限轨迹的完成数／观察跨度，不是持续稳态吞吐。原生 KV allocator 轨迹记录的是池已用块，不是请求持有的 `B_live`。三组 Talker 都曾接近或达到各自 KV 池上限；Thinker 每 rank 峰值约 47%。只有单次探索性实测，且增加 Talker 副本同时增加算力、执行名额和独立 KV 容量。尚未固定 SLO 契约，因此没有 SLO goodput。

[完整报告及图](/home/zhongyu/research_reports/kv_resource_ratios_20261027/ratio_4x_extended_900s/results.md) · [P95 延迟与被动抢占请求比例图](/home/zhongyu/research_reports/kv_resource_ratios_20261027/ratio_4x_extended_900s/latency_p95_and_preemptions.png) · [跨组统计 CSV](/home/zhongyu/research_reports/kv_resource_ratios_20261027/ratio_4x_extended_900s/results_summary.csv) · [可执行复现说明](README_RATIO_4X_EXTENDED.md)
