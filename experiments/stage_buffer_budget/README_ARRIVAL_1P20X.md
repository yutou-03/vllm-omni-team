# `mixed_pressure` 1.2× 到达密度扩展

原始 [ServeGen 多模态轨迹](/home/zhongyu/research_reports/servegen_active_buffer_20261012/mixed_pressure.jsonl)包含 70 个请求。该扩展只将每个 `arrival_offset_s` 除以 1.2；请求 ID、输入、媒体、采样参数、顺序均保持不变。原 120 秒注入窗口变为 100 秒，计划平均到达率从 0.5833 增至 0.7000 请求/秒。它是**派生压力负载**，不能用来估计原始轨迹下的自然抢占发生率，也不预设加压后必须出现抢占。

生成命令：

```bash
cd /home/zhongyu/project/vllm-omni-buffer
python3 experiments/stage_buffer_budget/derive_arrival_scaled_workload.py \
  --source /home/zhongyu/research_reports/servegen_active_buffer_20261012/mixed_pressure.jsonl \
  --output experiments/stage_buffer_budget/workloads/mixed_pressure_arrival_1p20x.jsonl \
  --factor 1.2 --source-window-s 120
```

派生文件及旁边的 `.manifest.json` 固定了两个 SHA-256、变换公式和负载统计。发送前需确认目标 GPU 空闲并启动**当前固定分支**的服务，例如：

```bash
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/launch_server.py \
  --session mixed_pressure_1p20x_20261019_gpu \
  --framework /home/zhongyu/project/vllm-omni-buffer \
  --deploy /home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget/deploy_urgent_keep.yaml \
  --extra-env-file /home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget/env.json
```

模型就绪后，用原研究已验证的 SSE 客户端开放环路发送；新入口不修改历史请求配额账本：

```bash
python3 experiments/stage_buffer_budget/replay_open_loop.py \
  --workload experiments/stage_buffer_budget/workloads/mixed_pressure_arrival_1p20x.jsonl \
  --output-dir /home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_1p20x_20261019 \
  --session-dir /home/zhongyu/research_reports/servegen_active_buffer_20261012/sessions/mixed_pressure_1p20x_20261019_gpu \
  --duration-s 100
```

结束时在容器内使用原研究的 `stop_server.py --session mixed_pressure_1p20x_20261019_gpu`。`results.json` 保留每个请求的实际发送偏差、完成状态和超时；240 秒是客户端安全终止上限，不是预设 SLO。该轨迹没有当前 urgent-keep 策略需要的逐请求 `omni_scheduling`，因此此运行只用于 Talker 容量压力观察，不可称为紧急 keep 收益实验。
