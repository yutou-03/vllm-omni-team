# 历史试验：`mixed_pressure` 70 条／30 秒到达时间压缩

本文件记录的是旧版“只压缩时间”的实测；用户现在要求的“四倍请求数且四倍时间压缩”是 [280 条／30 秒新轨迹](README_280_IN_30S.md)。两者不能混称。

冻结轨迹保留原始 70 条真实多模态请求及其内容、顺序和采样参数，只将 `arrival_offset_s` 除以 4：120 秒计划注入窗口缩为 30 秒，计划平均到达率从 0.5833 增至 2.3333 请求/秒。轨迹文件为 [`workloads/mixed_pressure_arrival_4x.jsonl`](workloads/mixed_pressure_arrival_4x.jsonl)，相邻 manifest 记录输入与派生 SHA-256。运行时所用源码是提交 `9106d309b7acccf6990da53b57080d726eef65ac` 的干净 Git 工作树；固定工作树中另一个对话正在改动的控制器文件没有进入该实测。

本主机的模型服务依赖 `vllm_omni_privileged` 容器。先确认 GPU 0–5 可用，不要与其他服务同时占用这些卡。从固定工作树创建一次性的 Git 工作树，并使用唯一的会话和结果目录：

```bash
cd /home/zhongyu/project/vllm-omni-buffer
CLEAN_SRC=/home/zhongyu/.cache/kv_replay_worktrees/mixed_pressure_4x_replay
RUN_TAG=mixed_pressure_4x_replay_$(date +%Y%m%d_%H%M%S)
SESSION_DIR=/home/zhongyu/research_reports/servegen_active_buffer_20261012/sessions/$RUN_TAG
OUTPUT_DIR=/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/$RUN_TAG
git worktree add --detach "$CLEAN_SRC" 9106d309b7acccf6990da53b57080d726eef65ac
```

如需重新生成轨迹，以下命令使用仓库中真实的生成器；已冻结的文件存在且内容相同时它只核验，不会重写：

```bash
python3 "$CLEAN_SRC/experiments/stage_buffer_budget/derive_arrival_scaled_workload.py" \
  --source /home/zhongyu/research_reports/servegen_active_buffer_20261012/mixed_pressure.jsonl \
  --output "$CLEAN_SRC/experiments/stage_buffer_budget/workloads/mixed_pressure_arrival_4x.jsonl" \
  --factor 4 --source-window-s 120
```

启动服务，待 `http://127.0.0.1:8095/v1/models` 可访问后才开放环路发送；本次实测保留自然排空和原有内存安全处理：

```bash
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/launch_server.py \
  --session "$RUN_TAG" --framework "$CLEAN_SRC" \
  --deploy "$CLEAN_SRC/experiments/stage_buffer_budget/deploy_urgent_keep.yaml" \
  --extra-env-file "$CLEAN_SRC/experiments/stage_buffer_budget/env.json"
python3 "$CLEAN_SRC/experiments/stage_buffer_budget/replay_open_loop.py" \
  --workload "$CLEAN_SRC/experiments/stage_buffer_budget/workloads/mixed_pressure_arrival_4x.jsonl" \
  --output-dir "$OUTPUT_DIR" --session-dir "$SESSION_DIR" --duration-s 30
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/stop_server.py \
  --session "$RUN_TAG"
```

即使客户端异常退出，也要执行停止命令。客户端 240 秒超时及服务排空等待是安全终止条件，并非 SLO。原始轨迹没有 urgent 请求的 `omni_scheduling` 元数据；本实验只能测各阶段池压力与被动抢占，不能当作主动 keep 或阶段缓冲收益对照。它也没有预注册的 SLO 契约。实测图使用 allocator 的原生池占用，而非请求持有物理 KV 块并集 `B_live`。

对已经产生的结果复算（`THREE` 为先前 3× 单次运行）：

```bash
THREE=/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_3x_20261019
python3 experiments/stage_buffer_budget/analyze_pressure_trace.py "$OUTPUT_DIR" --window-s 30 --output "$OUTPUT_DIR/pressure_injection.json"
python3 experiments/stage_buffer_budget/analyze_pressure_trace.py "$OUTPUT_DIR" --output "$OUTPUT_DIR/pressure_full.json"
python3 experiments/stage_buffer_budget/plot_stage_pool_pressure.py "$OUTPUT_DIR"
python3 experiments/stage_buffer_budget/compare_pressure_runs.py "$THREE" "$OUTPUT_DIR" \
  --first-label '3× arrival density' --second-label '4× arrival density' \
  --output-prefix "$OUTPUT_DIR/talker_pool_3x_vs_4x"
```

[2026-10-19 单次实测报告](/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_4x_20261019/results.md)列出原始数据、运行配置、两次真实被动抢占、统计口径与限制。该报告是压力观察；不是 FCFS/阶段缓冲配对因果实验。
