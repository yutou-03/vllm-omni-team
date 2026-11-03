# 实验二：阶段缓冲区设计，机制验证状态（2026-10-19）

## 研究边界与当前判定

原始导图 `/home/zhongyu/面向SLO感知的多模态流式推理调度优化.mm` 的“实验方案 → 实验二”要求把等待安全暂停点、暂停时长和恢复代价纳入决策；TBT 分支提出阶段间库存差异及保留 KV 的短暂停。本轮只验证保留 KV，不启用重算或换出。基线语义沿用 `multistage-scheduling-experiments-lowqps-v2/docs/experiment_plan_v2.md` 中的相同硬件、不可变负载、客户端完整消费终点和排空要求。实验一的阶段余量排序不进入任何对照。

**判定：真实 GPU 上尚未形成“暂停申请 → 紧急请求实际入批 → 原请求释放并重新执行”的事件链。当前数据只能支持库存与机制失败分析，不能报告策略性能收益，也不能执行正式三策略性能比较。**

## 既有 1.2× 与 3× 压力运行审计

可复算脚本为 `audit_stage_buffer_trace.py`，结果在 `/home/zhongyu/research_reports/stage_buffer_experiment_20261019/audit_1p20x_20261019.json` 和 `audit_3x_20261019.json`。统计按 `(logical_request_id, pipeline_attempt_id, stage_attempt_id, chunk_id)` 去重；发送、接收和开始消费以相同尝试的相同 chunk 核对，不能把已发布但尚未收到的 chunk 算入安全库存。Stage 0 单位是实际 embedding conditioning 位置（首包减去 prompt 行，再减去已选择位置），Stage 1 单位是新增 codec 帧（不重复计入 left context，开始转换即视为消费）。

| 运行 | Stage 0 发送/接收 chunk | Stage 1 发送/接收/开始转换 chunk | 守恒异常 | 真实主动暂停 |
| --- | ---: | ---: | --- | ---: |
| 1.2× | 4129/4129 | 726/726/726 | 0 | 0 |
| 3× | 4151/4129 | 717/717/717 | 0；旧尝试 0051 在被动抢占前发布、未被接收的 22 个 1 行 chunk，重启尝试单独计数 | 0 |

两个运行的 Stage 0 决策时 `Q_ready` 最大均为 1 位置，按保守消费速率 28.8517 位置/秒仅覆盖 **34.66 ms**；Stage 1 最大为 25 帧，按 1760.7870 帧/秒仅覆盖 **14.20 ms**。当前配置基础暂停高水位分别为 170 ms、50 ms，单是达到最小触发水位就至少需要 5 个 Stage 0 位置或 89 帧 Stage 1 库存；导图建议的固定 300/100 ms 需要更多。动态水位还取 `max(基础低水位, 最近 64 个忙调度间隔 P95 + 10 ms)`，在压力下只能更高。3× 的一个同一决策中，Stage 0 已发布库存为 200 位置，但接收端仅有 1 位置；1.2× 的对应最大已发布库存 143 时，接收端是 0。已发布量不能资助暂停。

按有非空批次的相邻调度起点间隔计算，1.2× Stage 0 P95/P99/最大为 86.62/905.95/4402.31 ms，3× 为 458.20/4981.35/10215.68 ms；3× Stage 1 为 116.28/213.14/748.06 ms。这些间隔包含计算、输出处理和调度等待，是下一安全检查点的墙钟风险代理，不能解释为纯 GPU kernel 时间。100 ms 只是本次 hold 的**申请上限**，同步批次运行中无法强制按时释放；释放之后还可能排队才重新入批。

## 带 SLO 的真实 GPU 机制烟测

冻结输入位于 `/home/zhongyu/research_reports/stage_buffer_experiment_20261019/frozen_workloads/`。两个 JSONL 都由真实图像加文本输入生成，包含 schema 1 的 `source_request_id`、`ingress_order`、`slo_ms`、`request_path=audio` 和正数 `predicted_stage_ms=[3000,5000,1000]`。SLO 120 s/9 s 是机制烟测的预声明触发条件，**不是经独立校准的正式性能 SLO**。

- 三请求提前量烟测：`stage0_slo_smoke`，负载 SHA-256 `be48127f264c43876b7f9b344dac03f0183502e7d4c47fa38a8d17055317c58f`，0/6/9 s 到达，3/3 客户端成功并排空；1 次候选受害者 decode 边界成立，但 `Q_ready=0`，其余两次检查执行名额已空闲。事件链四项计数均为 0。
- 两请求重叠烟测：`stage0_overlap_slo_smoke`，负载 SHA-256 `ecd1d48c6fb28f7265d655fa2701249331c64870ae188436cb567f6ab2f73539`，0/3 s 到达，2/2 客户端成功并排空；10 次候选检查中受害者均在 decode 边界，但 `Q_ready=0`、高水位 170 ms，全部因 `insufficient_downstream_coverage` 拒绝。事件链四项计数均为 0。

客户端成功与 chunk 守恒只能证明这些未暂停运行正常完成，**不能证明暂停后无丢失或重复**。`stage0_slo_smoke/experiment_manifest.json` 和 `stage0_overlap_slo_smoke/experiment_manifest.json` 保存各自的模型、GPU 映射、K=1/R=64、KV 池、token budget、源代码哈希、配置哈希、负载哈希、客户端结果与事件 trace 哈希；原始 SSE 和压缩前 JSONL 留在各 run 目录。部署使用 Stage 0 GPU 0–3 TP4、Stage 1 GPU 4、Stage 2 GPU 5，Stage 0/1 的单轮 token budget 均为 32768。服务会话在烟测后已停止。

## 实现审计和修复

GPU 会话运行的是框架提交 `8c7a291e8c6df489101471ef432bb2cad45b3d05`，原 `stage_buffer_controller_budget.py` 哈希已在启动 manifest 内。审计发现控制器按逻辑请求 ID 累计库存；3× 运行中的请求 0051 被动重启时物理执行 ID 变为 `__execution1`、chunk ID 从零开始，逻辑 ID 不变。因此旧/新尝试可能混用库存，且新物理 ID 查不到自己的库存。

烟测结束后已改为按**物理执行 ID**索引库存，并在同一物理 ID 出现不同尝试编号时失效关闭；新增 CPU 回归覆盖旧尝试和重启尝试隔离。`verify_budget_policy_cpu.py` 的 19 项检查通过。补丁保存在 `/home/zhongyu/research_reports/stage_buffer_experiment_20261019/post_smoke_inventory_fix.patch`，该修复**尚未重新在 GPU 上验证**，不能倒填进上述烟测的部署版本。

## 下一步可验证的改进与正式实验门槛

当前首先缺的是接收端真实可用库存，随后才是同步调度检查点的时长上界。下一实现应在接收端提供有容量和顺序保证的 bounded lookahead 队列，并分别记录 `sent → received → model-input-selected / conversion-started` 的 chunk 身份、时间和容量；只从已接收、尚未取用的同一尝试数据给暂停记账。队列长度至少要能够跨过上述安全高水位，且要用实测生产/消费速率检验，不以调低阈值制造触发。若排队放大内存或端到端延迟，也必须作为代价计入。另需控制最长同步执行切片，或建立可信的最坏/高分位检查点等待上界；只调整 100 ms 申请参数无效。

实现后的**机制门**：使用冻结、合法 SLO 的负载，在真实 GPU 上同时看到 `small_hold_requested`、匹配受害者的 `urgent_keep_admitted`（确认受害者未同轮入批且保留其 KV）、`small_hold_released`、`small_hold_first_rescheduled` 和受害者恢复后第一个新 chunk；逐尝试核对 chunk ID 连续、发送/接收/消费守恒、客户端输出完整，并记录暂停申请到释放、到重新入批及恢复首 chunk 的墙钟时间。只要任一环缺失，继续标为机制试跑。

机制门通过后，才冻结正式三臂：无主动暂停、按阶段固定水位（从导图 300/100 ms 起并保留恢复安全预算）、加入 checkpoint 等待与恢复代价的动态水位。三臂必须使用相同 `stage_deadline_edf_np` 等待顺序、模型、GPU/阶段映射、K/R、KV 驻留上限、token budget、SLO 契约、到达 JSONL 和种子；关闭实验一余量排序。当前紧急模式把目标阶段 EDF 绑定在开关上，正式比较前需把“等待顺序”和“是否允许暂停”拆成独立配置，避免对照污染。每个种子随机化运行顺序，先 warmup 并排空；保存完整客户端 SSE、服务端事件、drain 证明和哈希清单。

主指标按客户端完整消费响应后、以全部 offered 请求为分母计算按期完成率及 goodput；同时分层报告紧急请求 SLO、受害者输出断流/尾延迟、实际暂停超期、释放到重新入批和恢复首 chunk、KV 峰值与被动抢占。需要共同负载点和多个配对种子，不能以本次 2–3 请求烟测计算性能收益。导图“超过 512 token 后预测更准”是待验证假设：当前 3× 负载 70 条中只有 1 条 `max_tokens>512`，而实测文本输出最长仅 230 token，跨越 512 的样本数为 0；正式工作负载需在相同路径和输入规模分层下有足够实际跨过 512 的样本，用独立校准/验证集比较前后预测误差（MAPE、P95 绝对误差和低估尾部），并报告置信区间与批大小条件。
