# `mixed_pressure` 3× 到达密度压力运行

这条冻结轨迹沿用原始 ServeGen `mixed_pressure.jsonl` 的 70 个真实多模态请求（video 23、audio 23、image 24）。仅将每个 `arrival_offset_s` 除以 3；请求 ID、内容、媒体、采样参数和顺序均不变。计划注入窗口从 120 秒缩为 40 秒，平均计划到达率从 0.5833 增至 1.7500 请求/秒。轨迹及相邻 manifest 记录源和派生 SHA-256。它是人为加密的压力轨迹，不能代表原始自然负载中内存不足的发生频率。

```bash
cd /home/zhongyu/project/vllm-omni-buffer
python3 experiments/stage_buffer_budget/derive_arrival_scaled_workload.py \
  --source /home/zhongyu/research_reports/servegen_active_buffer_20261012/mixed_pressure.jsonl \
  --output experiments/stage_buffer_budget/workloads/mixed_pressure_arrival_3x.jsonl \
  --factor 3 --source-window-s 120
```

该主机的服务依赖位于 `vllm_omni_privileged` 容器内。确认 GPU 0–5 空闲后，用固定工作树、部署文件和环境启动；不同会话使用不同名称和空输出目录：

```bash
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/launch_server.py \
  --session mixed_pressure_3x_20261019_gpu \
  --framework /home/zhongyu/project/vllm-omni-buffer \
  --deploy /home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget/deploy_urgent_keep.yaml \
  --extra-env-file /home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget/env.json
```

待 `http://127.0.0.1:8095/v1/models` 就绪后，从宿主机开放环路发送，并保留所有客户端状态、发送偏差、服务器 trace 和排空状态：

```bash
python3 experiments/stage_buffer_budget/replay_open_loop.py \
  --workload experiments/stage_buffer_budget/workloads/mixed_pressure_arrival_3x.jsonl \
  --output-dir /home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/mixed_pressure_3x_20261019 \
  --session-dir /home/zhongyu/research_reports/servegen_active_buffer_20261012/sessions/mixed_pressure_3x_20261019_gpu \
  --duration-s 40
```

```bash
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/stop_server.py \
  --session mixed_pressure_3x_20261019_gpu
```

客户端的 240 秒超时和服务排空的 90 秒等待是安全终止条件，不是 SLO。发送端复用原实验的音频右补齐审计，并校验源轨迹 SHA；这只修正客户端预期的 prompt token 数，不改请求内容。运行保留服务端内存安全队列和被动抢占。本轨迹没有 urgent 请求的 `omni_scheduling` 元数据，因此仅用于检查阶段池压力及其后果，不能称为主动抢占或阶段缓冲收益实验。
