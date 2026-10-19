# 3× `mixed_pressure` 单次压力观察

冻结轨迹保留原始 70 条多模态请求，仅把到达时刻除以 3；40 秒内开放环路发送，70/70 条完成。Talker（GPU 4，1.894 GB KV 池）在排空期到达 0 个空闲块，发生 1 次真实 KV 分配失败与被动抢占；受害者释放 83 块，随后重启 Thinker、Talker、Code2Wav 三阶段，整条链追赶旧进度耗时约 41.4 秒。Thinker 每 rank 已观测 KV 峰值为 27.14%，但其批次曾达到 64 条/32768-token 执行上限，不能由低 KV 占用推断算力富余。

[完整实测报告](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_3x_20261019/results.md)包含 [完整阶段池图](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_3x_20261019/pool_occupancy_full_3x.png)、[抢占时空闲块图](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_3x_20261019/talker_free_blocks_preemption_detail_3x.png)、原始日志、配置、文件哈希和复算命令。此运行没有主动 keep hold，也没有 SLO 契约或阶段队列对照；它证明高压下**存在**局部池耗尽及跨阶段重启，尚不能证明发生频率或阶段缓冲收益。
