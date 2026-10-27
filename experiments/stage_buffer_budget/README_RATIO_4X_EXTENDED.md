# 更正实验：放宽 240 秒主动取消，记录自然完成跨度

原 [`README_RATIO_4X.md`](README_RATIO_4X.md) 的三组回放使用了归档客户端的 240 秒每请求 watchdog。到时客户端关闭 SSE 连接，服务端相应取消请求；旧结果 179／270／277 仅表示这一人为截止下的完成数，不能解释为无截止自然完成量，也不能把 270 秒曲线尾部当成自然排空。旧数据与图保留，并在报告开头标注更正。

此更正实验仍使用同一 `mixed_pressure_280_in_30s.jsonl`（SHA-256 `5c3b04742649a4bd694a46b7850fab6e8fd6e9d40cb50c4a1e6edca867197693`）、同一模型、三种配置及相同 30 秒外生到达表。服务与客户端源码固定在提交 `2a90b2d5ebe6e8dc10008b8026d7a18955d672d7` 的干净工作树。新增的 `sse_client_extended.py` 是归档客户端的逐字副本，唯一行为变动是允许每请求 watchdog 最大 900 秒；`replay_open_loop.py` 从该 Git 跟踪文件导入 `send_request`。900 秒仍是安全终止条件，不是 SLO。如果任何请求触及它，应按截尾报告，不能称该组自然排空。

三组依次运行；Thinker TP4 固定 GPU 0–3，Code2Wav 固定 GPU 5，Talker TP1 独立副本使用 GPU 4／4+6／4+6+7。每次独占所需 GPU，服务预热和图捕获结束、`/v1/models` 返回后才开始回放。服务端代码、KV 池容量、block size、调度策略、请求内容与采样参数均不变；仅客户端安全上限从 240 扩至 900 秒。两次实验结果目录必须分开。

设备映射注意：部署 YAML 的 `stages[].devices` 是 `CUDA_VISIBLE_DEVICES` 列表中的逻辑序号。例如物理 GPU 6 被其他作业占用时，可将 4:2 的 `--devices` 改为 `0,1,2,3,4,5,7`，**部署 YAML 仍用原文件的 `devices: "4,6"`**；此时逻辑 6 映射到物理 GPU 7。须在会话 manifest 中保留实际可见设备列表和 GPU PID 映射。不能直接在 YAML 写 `4,7`，因为七张可见 GPU 的逻辑序号只到 6。

```bash
cd /home/zhongyu/project/vllm-omni-buffer
SRC=/home/zhongyu/.cache/kv_replay_worktrees/ratio_4x_2a90b2
# 首次：git worktree add --detach "$SRC" 2a90b2d5ebe6e8dc10008b8026d7a18955d672d7
X=1  # 顺序运行 X=1、2、3
DEVICES=0,1,2,3,4,5  # X=2 时加 6；X=3 时加 6,7
TAG=ratio4${X}_extended_$(date +%Y%m%d_%H%M%S)
SESSION=/home/zhongyu/research_reports/servegen_active_buffer_20261012/sessions/$TAG
BASE=/home/zhongyu/research_reports/kv_resource_ratios_20261027/ratio_4x_extended_900s
OUT=$BASE/ratio_4_${X}_run1
mkdir -p "$BASE"
docker exec vllm_omni_privileged python3 \
  "$SRC/experiments/stage_buffer_budget/launch_ratio_server.py" \
  --session "$TAG" --framework "$SRC" \
  --deploy "$SRC/experiments/stage_buffer_budget/deploy_ratio_4_${X}.yaml" \
  --devices "$DEVICES" \
  --extra-env-file "$SRC/experiments/stage_buffer_budget/env.json"
# 等待服务就绪，然后执行；建议外层 shell trap 确保异常时也停止服务：
docker exec vllm_omni_privileged python3 \
  "$SRC/experiments/stage_buffer_budget/replay_open_loop.py" \
  --workload "$SRC/experiments/stage_buffer_budget/workloads/mixed_pressure_280_in_30s.jsonl" \
  --output-dir "$OUT" --session-dir "$SESSION" --duration-s 30 --timeout-s 900
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/stop_server.py \
  --session "$TAG"
```

主指标为全体 280 条无超时完成时的实际排空跨度，以及 `280 / 排空跨度`；同时记录请求原始 TTFT、TTFP、E2E 和客户端接受的音频内容／排空跨度。若仍有截尾，原始延迟保留空值及原因，并仅报告“900 秒约束下”的完成率。原生 allocator KV 池轨迹继续分别按 GPU 记录，不把它称为请求持有 `B_live`。绘图对齐相同回放零点，展示至三组中最晚的实际观察终点，不将安全上限后的空闲时间填入吞吐分母。

分析、`run_manifest.json`、GPU PID 映射与 PNG/PDF 命令同 [`README_RATIO_4X.md`](README_RATIO_4X.md) 的后半部分，唯一不同是本轮结果目录 `$BASE`、源码提交和 900 秒参数。代码会从 `protocol.json` 读取实际超时设置，不再写死 270 秒。三组均做 1 次探索性实测，不能用时间采样点充当重复实验。
