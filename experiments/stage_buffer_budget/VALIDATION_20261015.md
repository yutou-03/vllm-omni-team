# 2026-10-15 紧急 keep 准入验证边界

本页记录功能提交 `b8f315c194ef1277eba81e17f8ff0a629d41513d` 的真实请求烟测。模型是 `/data/models/Qwen3-Omni-30B-A3B-Instruct`，Stage 0 使用 GPU 0–3、TP4、每 rank 14,136,115,200 byte KV 池；Stage 1 使用 GPU 4、1,894,318,080 byte KV 池。功能只在 Stage 0 打开，执行名额 K=1、驻留上限 R=64，100 ms hold 上限及库存保护保持启用。其余阶段保持原生 FCFS。所有请求均来自现有 `servegen_profile_20261011/image_high.jsonl` 的真实图像+文本输入，输出路径经过 Talker。正式性能比较尚未运行。

| 运行 | 外生到达（秒） | 负载 SHA256 | 完成 | 主动 hold / 紧急成功准入 | 关键观察 |
| --- | --- | --- | --- | --- | --- |
| `urgent_keep_stage0_20261015` | 0, 10 | `37c7d1737a09392e327cf6fb38a8a4b47346535647c2a7f12ef576724a321a70` | 2/2 | 0 / 0 | 背景 Thinker 在约第 6 秒完成，紧急请求到达时 K 已空。 |
| `urgent_keep_stage0_overlap_20261015` | 0, 3 | `ecd1d48c6fb28f7265d655fa2701249331c64870ae188436cb567f6ab2f73539` | 2/2 | 0 / 0 | 有阶段重叠；紧急到达后 victim 仍持有 324 个物理 KV 块，接收端安全库存仅 0–1 个位置。 |
| `urgent_keep_stage0_three_20261015` | 0, 6, 9 | `0e25472734b9905f489f0bd0a37903456b3001ea6612ad9c3d1a3784100fb8` | 3/3 | 0 / 0 | 增加一条先行长请求仍未让目标 victim 的接收端安全库存超过 1 个位置。 |

后两次各有 10 次节流后的 `no_eligible_victim` 决策。离线按同一 `chunk_id` 去重，从发布/接收的 embedding 行数扣除 prompt 和 Talker 已选择位置，得到每次决策时 `Q_ready` 为 0–1；控制器的 Stage 0 暂停高水位是 170 ms，因而拒绝是当前保护条件下的预期行为。Stage 1 在三请求运行中单请求的瞬时已接收、未开始转换 codec 库存最大 25 帧；现有 CPU 标定测试同样判定 25 帧不足以安全暂停。没有数据支持声称本次 GPU 运行已完成暂停、保留 KV 后准入和恢复。

原始证据位于 `/home/zhongyu/research_reports/servegen_active_buffer_20261012/runs/<运行名>/`，包括 `protocol.json`、`results.json` 和裁剪后的 `traces/stage_events.jsonl`。服务清单位于同项目的 `sessions/urgent_keep_stage0_20261015/manifest.json`，记录了启动时的全部 Python 源文件 hash、部署配置 hash 和 GPU 预检。三次回放均排空成功，服务已在原容器中停止，GPU 0–5 均释放。

CPU 验证覆盖“hold 时 victim 不参与 batch、仍持有 KV、urgent 实际入批后才计成功、victim 首次重新入批才算恢复”，以及 R 已满、deadline 非倒置、库存不足、超时停用等保护。最新诊断补丁新增每个 victim 的具体拒绝理由，仅完成 CPU 验证；上面三个 GPU 运行使用的是 `b8f315c1`，其事件只有笼统的 `no_eligible_victim`。下一步需要在不放宽保护的条件下，找到或标定有足够 receiver-ready 库存且 K 已满、R 未满的负载，再做实际状态转换验证和 FCFS 成对评价。
