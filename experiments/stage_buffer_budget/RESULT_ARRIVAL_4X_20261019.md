# 4× `mixed_pressure` 单次压力观察

原轨迹 70 条多模态请求仅压缩到达时间，30 秒开放环路注入；70/70 条最终成功，0 客户端超时。服务运行在干净提交 `9106d309`，固定工作树中另一对话的未提交控制器修改未进入实验。与 3× 运行相比，641 个运行时 Python 文件的哈希相同。

Talker（GPU 4，1.894 GB 独立 KV 池）的原生占用峰值为 100%，先后发生 **2 次真实 KV 分配失败、2 次被动抢占**，分别释放 331 和 173 块，并分别触发 Thinker、Talker、Code2Wav 的整链重启。被废弃的已计算进度为 5283 和 2768 tokens；重启到恢复完成的 85.982 和 43.211 秒包含等待与重新生成，不是纯 GPU 重算时间。485 条容量检查失败是对 9 个根请求的重复探测，不是 485 次抢占。4× 的 Talker 高占用和两次抢占都发生于 30 秒注入结束后的排空阶段，不能据此推断到达率与瞬时占用线性相关。3× 和 4× 最终交付的 70 条请求 payload、文本和音频 chunk 哈希均一致。

[完整报告](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_4x_20261019/results.md)附有 [4× 阶段池图](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_4x_20261019/pool_occupancy_full_4x.png)、[3×/4× Talker 对照图](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_4x_20261019/talker_pool_3x_vs_4x.png)、[首次被动抢占细节](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_4x_20261019/talker_free_blocks_preemption_1_4x.png)和[第二次细节](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_4x_20261019/talker_free_blocks_preemption_2_4x.png)。原始服务日志、客户端逐请求记录、运行 manifest 和复算脚本也保留在报告目录。复跑命令见 [README_ARRIVAL_4X.md](README_ARRIVAL_4X.md)。

本次没有接受主动 keep hold，未测 `B_paused`；原生池占用百分比也不是请求持有 KV `B_live`。这是每个密度各一次的非配对压力观察，没有预注册 SLO 或阶段队列对照，不能据此宣称阶段缓冲改善了吞吐或达标率。

[3×/4× 到达传播诊断](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_4x_20261019/frequency_diagnosis_20261026.md)和[实测图](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_4x_20261019/arrival_to_talker_3x_4x.png)解释客户端加速为何只带来 1→2 次真实抢占。
