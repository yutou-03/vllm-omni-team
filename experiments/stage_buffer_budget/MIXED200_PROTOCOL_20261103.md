# 200 请求混合路径对照

每条轨迹 200 条：120 文本、80 音频，各路径一半标准 SLO、一半宽松 SLO。标准/宽松输出上限：文本 128/512，音频伴随文本 64/128；不强制忽略 EOS。两策略相同请求与种子。

独立低负载检查 8 条（每层 2 条，另有 2 条 warmup），只作为 pilot 标定。当前文本 TTFT 标准/宽松 1.3/5.2 s；音频首段 3/6 s。文本完整交付期限 45/160 s，音频 90/180 s。文本平均可见 TPOT 上限 1/4.8 s，最大阅读断流 0.1 s、累计 0.5 s；音频最大单次断流 0.25 s、累计 0.5 s。音频伴随文本 TTFT 另检查 2.6/5.2 s。

每条到达窗口 166.667 s，平均 1.2 请求/s。泊松采用固定窗口内恰好 200 次到达的条件泊松过程（均匀次序统计量）。BurstGPT 使用官方 CSV 中 ChatGPT/API 成功记录：稳定时间排序、非重叠 200 条分块，取间隔 CV 的第 75 百分位块，统一缩放时间；选择发生在策略比较之前。只复用到达形状，不声称合成内容、路径或 SLO 来自 BurstGPT。

原生组为 native FIFO、相同 K16/R64 与日志配置，无新增接收队列、文本放行或主动暂停。这是资源对齐的原生对照，不是默认 K64 吞吐。方案组在 Stage 0/1 使用独立暂停状态，保留 KV、FIFO，启用文本库存、Q01 接收队列及 Q12 有界放行。Q12 放行 30 codec frame/s，库存计入前仍扣除一个在途 chunk；真实安全覆盖不足时不暂停。GPU 机制验证前不能声称双阶段暂停成立。

先跑单独 24 请求混合烟测，要求真实暂停、紧急入批、恢复新输出、KV 保留和输出守恒。然后优先 ours/native BurstGPT，再 native/ours Poisson；这不是多种子随机重复。每轮服务独立启动，统一预热，两路径排空。仅一个 worker 容器，结束自动删除。

## 指标

- Gstream：沿用 soft_ttft_v2，β=μ=3.3，以固定到达窗口 T 归一化。另报去掉库存权重后的敏感性和分项，不能把位置敏感的分数改善归因为模型吞吐。
- Generated tokens/s：Thinker 实际生成的 model token IDs，含音频路径伴随文本，计一次；除以最早计划到达至最后 Thinker 生成事件的时长。不是三阶段所有内部 token 的总和。
- SLO tokens/s：完整且满足该请求全部预设路径 SLO 的生成 token 数，使用同一生成时长。
- 另报交付 tokens/s 与可播放音频秒/s，分母为最早到达至最后客户端完成。这样能看见末端 pacing 的真实成本。
- Mean TTFT、TTFT CDF：全体、文本、音频分别报告。CDF 的分母含全部到达请求，无首输出请求不能从分母删除；均值说明有效样本数。音频 TTFP 另报。
- 生产 TPOT、交付 TPOT、播放断流、完整率、暂停次数/完整链、实际暂停及恢复首输出时间。

200 个客户端 worker 独立按时发送，不受旧 64 worker 池限制；记录实际发送偏差。反馈文件限频 0.2 s，原始 SSE、音频与每事件时刻完整保留。调度不使用这个客户端反馈文件作库存信用。

## 复现入口与证据

结果根目录：/home/zhongyu/research_reports/stage_buffer_mixed200_20261103/

源码：mixed200_service.py、calibrate_mixed200.py、prepare_mixed200.py、run_mixed200.py、analyze_mixed200.py。

保留 worker/manifest.json 中实际服务源码逐文件哈希与镜像 ID；测量代码的提交另外记录。运行前服务快照仅更新允许两个独立阶段的入口验证，记录在 manifest；没有覆盖其他人的修改。已有 profile_allocations.py 工作区修改未纳入本次提交，服务快照保留其实际哈希。

BurstGPT 来源：https://github.com/HPMLL/BurstGPT 。源 CSV、选中记录、原行号、SHA256、缩放比例均在结果 sources/ 中。
