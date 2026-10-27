# `mixed_pressure` 四倍请求数、四倍时间压缩：280 条／30 秒

本实验从冻结的 70 条、120 秒多模态轨迹生成 4 个请求副本，每个副本使用唯一的 `request_id` 与 `source_request_id`，其余输入内容、模态和采样参数保持不变；每份的计划到达时刻都是原时刻除以 4。**280 条请求在 30 秒注入，计划平均速率 9.333 req/s，为原轨迹的 16 倍。** 四份同源请求在相同计划时刻发送，因此微突发也增强。它是“70 种内容各重复 4 次”的受控压力负载，不是 280 种互不相同的内容，也不能代表自然生产流量。历史的 `mixed_pressure_arrival_4x.jsonl` 是 70 条／30 秒的到达时间压缩试验，文件和结果均不覆盖。

轨迹和 manifest 在 [`workloads/mixed_pressure_280_in_30s.jsonl`](workloads/mixed_pressure_280_in_30s.jsonl) 及同目录 `.manifest.json`。生成器拒绝不同内容覆盖，客户端重新核对源 SHA-256、每份请求内容、复制索引、到达时刻、唯一 ID，并把 23 个原始音频校验记录映射到 92 个音频副本。服务端跨请求 prefix caching 在前次运行的启动日志中实际为关闭；本次也必须再次核对。没有 urgent `omni_scheduling` 元数据，因此只能研究池压力与被动抢占，不能当作主动 keep 或阶段缓冲收益对照。

```bash
cd /home/zhongyu/project/vllm-omni-buffer
python3 experiments/stage_buffer_budget/derive_replicated_workload.py \
  --source /home/zhongyu/research_reports/servegen_active_buffer_20261012/mixed_pressure.jsonl \
  --output experiments/stage_buffer_budget/workloads/mixed_pressure_280_in_30s.jsonl \
  --request-factor 4 --time-factor 4 --source-window-s 120
```

本机服务使用 `vllm_omni_privileged` 容器。确认 GPU 0–5 空闲，先将实验代码与轨迹提交，再由该提交创建干净 Git 工作树运行；固定工作树里另一对话的未提交控制器改动不能混入。本段的 `COMMIT` 应填实际冻结的 Git SHA；每次使用唯一运行名和空目录。

```bash
cd /home/zhongyu/project/vllm-omni-buffer
COMMIT=<冻结的实验提交SHA>
CLEAN_SRC=/home/zhongyu/.cache/kv_replay_worktrees/mixed_pressure_280_in_30s
RUN_TAG=mixed_pressure_280_in_30s_$(date +%Y%m%d_%H%M%S)
SESSION_DIR=/home/zhongyu/research_reports/servegen_active_buffer_20261012/sessions/$RUN_TAG
OUTPUT_DIR=/home/zhongyu/research_reports/kv_residency_trace_20261013/stage_pool_evidence/$RUN_TAG
git worktree add --detach "$CLEAN_SRC" "$COMMIT"
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/launch_server.py \
  --session "$RUN_TAG" --framework "$CLEAN_SRC" \
  --deploy "$CLEAN_SRC/experiments/stage_buffer_budget/deploy_urgent_keep.yaml" \
  --extra-env-file "$CLEAN_SRC/experiments/stage_buffer_budget/env.json"
```

待 `http://127.0.0.1:8095/v1/models` 就绪后运行开放环路回放，完成或异常时均停止服务并保存日志：

```bash
python3 "$CLEAN_SRC/experiments/stage_buffer_budget/replay_open_loop.py" \
  --workload "$CLEAN_SRC/experiments/stage_buffer_budget/workloads/mixed_pressure_280_in_30s.jsonl" \
  --output-dir "$OUTPUT_DIR" --session-dir "$SESSION_DIR" --duration-s 30
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/stop_server.py \
  --session "$RUN_TAG"
```

客户端每请求 240 秒是原有安全终止限制，不是 SLO；280 条可能在截止前无法全部完成，超时和错误必须保留在分母。不要仅分析成功请求。报告实际发送偏差、目标阶段真实输入就绪与排空、KV 池占用、真实 `allocate_slots` 失败和 victim 事件、丢弃进度及恢复、有效输出。原生池占用与不可重新分配的请求持有 KV `B_live` 不同。该压力运行不能估计自然流量的被动抢占发生率，也不能单独证明阶段缓冲收益。
