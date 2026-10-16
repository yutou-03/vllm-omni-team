# Stage 1 紧急请求 keep 准入（实验开关）

`env_urgent_keep_stage1.json` 只在 Stage 1 打开执行名额 K=2，`max_num_seqs=64` 仍是驻留名额 R；Stage 0/2 继续使用原调度。Stage 1 等待队列在初始化时由原生 FCFS 局部切换为 `stage_deadline_edf_np`。与原始库存缓冲策略不同，此模式仅在真正的阶段 deadline 倒置且 urgent slack 不足时申请一个 hold。victim 还须在确定的解码边界、仅允许每请求一次，并有足够的接收端库存预算；R 满则不暂停。请求仍占有 GPU KV，实际内存不足时保持原生安全排队/被动抢占。

等待请求必须提供合法 `omni_scheduling`，例如 schema 1 的 `source_request_id`、`ingress_order`、`slo_ms`、`request_path` 与三个阶段的正数 `predicted_stage_ms`。服务端生成单调时钟下的绝对阶段 deadline；客户端不能指定绝对值。历史 `canary.jsonl`/`mixed_pressure.jsonl` 没有该字段，不能直接用来声称触发紧急 keep。要用历史内容做新实验，需复制负载并明确增加 SLO 契约、固定到达时间和随机种子，保存新负载 hash。

新模式启动示例（仅启动服务，压测客户端需重放带 SLO 的工作负载）：

```bash
python3 /home/zhongyu/research_reports/servegen_active_buffer_20261012/launch_server.py \
  --session urgent_keep_stage1_trial \
  --framework /home/zhongyu/project/vllm-omni-buffer \
  --deploy /home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget/deploy_urgent_keep.yaml \
  --extra-env-file /home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget/env_urgent_keep_stage1.json
```

须在新 run manifest 中保存完整 Git commit、配置 hash、负载 hash 和实际调度事件。`small_hold_requested` 是申请；只有 `urgent_keep_admitted` 且 victim 未在同轮执行，才证明新请求利用了空位。`small_hold_first_rescheduled` 才是 victim 真正恢复。`urgent_keep_admission_pending` 表示未实际入批，不能算成功。超时的 hold 只在下次同步调度边界检查，100 ms 是申请上限，不保证实际暂停墙钟时间。 暂停期间不再覆盖单轮 token 预算；Stage 0/1 保持原配置 32,768，紧急请求 prefill 可能延后下次检查。

启动前 CPU 检查：`python3 experiments/stage_buffer_budget/verify_budget_policy_cpu.py`。正式评价必须另做实际两请求/多请求运行、KV 块归属对账及 FCFS 成对对照。此文档不是性能结论。

两请求真实输入烟测可从现有历史图像轨迹生成（仅验证转换，不作为性能实验）：

```bash
python3 experiments/stage_buffer_budget/make_urgent_keep_smoke.py \
  --source /home/zhongyu/research_reports/servegen_profile_20261011/image_high.jsonl \
  --output /tmp/urgent_keep_stage0_smoke.jsonl
```

该测试使用 Stage 0 的 K=1 配置、背景请求 120 s SLO、3 s 后到达的请求 9 s SLO；紧急请求较紧的 SLO 是预先声明的正确性触发条件。正式 Stage 1 实验需至少填满其 K=2 个执行名额，并使用另行冻结的外生工作负载。

首轮 GPU 标定显示原先第 10 秒到达太晚：背景 Thinker 约第 6 秒结束；现将默认到达改为第 3 秒，以检查同阶段重叠。也可用 `--lead-index 4 --background-delay-s 6 --urgent-delay-s 9` 生成三请求序列以检查下游积压是否自然出现。两种设置在 2026-10-15 烟测中均未满足接收端安全库存门槛，详见 [VALIDATION_20261015.md](VALIDATION_20261015.md)。
