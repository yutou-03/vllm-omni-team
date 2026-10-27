# Thinker:Talker = 4:1、4:2、4:3 的实测回放

使用冻结的 [`mixed_pressure_280_in_30s.jsonl`](workloads/mixed_pressure_280_in_30s.jsonl)：原 70 个多模态请求各复制四份，四份使用同一计划到达时刻（原时刻除以 4），总计 280 条、30 秒注入。三组的请求 ID、媒体、采样参数、到达表、模型和 Thinker TP4 完全相同。Talker 使用 GPU 4、4+6、4+6+7 上的 1、2、3 个独立 TP1 副本；Code2Wav 固定 GPU 5。每个 Talker 副本有自己的 KV 池，**不得把其他卡的空闲容量视为该副本可借用的容量**。

服务源码固定在提交 `9fb7ed5f5d7d89bc019d82b83e46d139b31d0174` 的干净工作树中，因此另一会话的未提交控制器改动不会混入。三组部署文件是 [`deploy_ratio_4_1.yaml`](deploy_ratio_4_1.yaml)、[`deploy_ratio_4_2.yaml`](deploy_ratio_4_2.yaml)、[`deploy_ratio_4_3.yaml`](deploy_ratio_4_3.yaml)。它们继承相同基础配置；Thinker 每 rank 配置 14,136,115,200 bytes KV，Talker 每副本配置 1,894,318,080 bytes KV，prefix cache 关闭，`max_num_seqs=64`，token budget 32,768。实际池容量及 GPU 映射需以每次运行的原生快照和清单核对。

本机服务依赖 `vllm_omni_privileged` 容器。三组必须顺序运行，不与其他实验共享 GPU。以下是单组命令，把 `X=1`、`2`、`3` 分别运行一次，且使用全新的会话名和结果目录。每次先用 `nvidia-smi` 确认所需 GPU 空闲；[启动器](launch_ratio_server.py)也会拒绝占用中的 GPU。启动后等 `/v1/models` 就绪、编译和图捕获完成，再执行回放。即使回放超时或失败，也要运行停止命令并保留原始文件。

```bash
cd /home/zhongyu/project/vllm-omni-buffer
SRC=/home/zhongyu/.cache/kv_replay_worktrees/ratio_4x_9fb7ed5
# 首次运行：git worktree add --detach "$SRC" 9fb7ed5f5d7d89bc019d82b83e46d139b31d0174
X=1
DEVICES=0,1,2,3,4,5  # X=2 用 0,1,2,3,4,5,6；X=3 再加 7
TAG=ratio4${X}_$(date +%Y%m%d_%H%M%S)
SESSION=/home/zhongyu/research_reports/servegen_active_buffer_20261012/sessions/$TAG
OUT=/home/zhongyu/research_reports/kv_resource_ratios_20261027/ratio_4x_280_in_30s/ratio_4_${X}_run1
docker exec vllm_omni_privileged python3 \
  "$SRC/experiments/stage_buffer_budget/launch_ratio_server.py" \
  --session "$TAG" --framework "$SRC" \
  --deploy "$SRC/experiments/stage_buffer_budget/deploy_ratio_4_${X}.yaml" \
  --devices "$DEVICES" \
  --extra-env-file "$SRC/experiments/stage_buffer_budget/env.json"
# 等待 http://127.0.0.1:8095/v1/models 返回模型，再运行：
docker exec vllm_omni_privileged python3 \
  "$SRC/experiments/stage_buffer_budget/replay_open_loop.py" \
  --workload "$SRC/experiments/stage_buffer_budget/workloads/mixed_pressure_280_in_30s.jsonl" \
  --output-dir "$OUT" --session-dir "$SESSION" --duration-s 30 --timeout-s 240
docker exec vllm_omni_privileged python3 \
  /home/zhongyu/research_reports/servegen_active_buffer_20261012/stop_server.py \
  --session "$TAG"
```

客户端 240 秒是安全截止，不是 SLO。用户尚未选定 SLO 契约，因而本轮不计算“满足 SLO 的 goodput”。我们报告固定注入加安全截止窗口（270 秒）内的成功完成请求数／秒、客户端实际接受的音频帧／秒及音频内容秒／墙钟秒，并附 TTFT、TTFP、E2E 原始延迟。未完成请求与错误保留在分母。此处 KV 主图来自 allocator 的物理池块占用，并非请求持有 KV 的 `B_live`；每个 Talker 副本和 Thinker TP rank 分别标注。图使用原始阶梯采样，不插值。

回放结束后，在宿主机使用 [`analyze_ratio_replay.py`](analyze_ratio_replay.py) 对每个 `$OUT` 生成 `kv_pools_by_engine.csv`、`request_summary.csv`、`ratio_summary.json` 和 PNG/PDF；Thinker TP4 每块 393,216 bytes，Talker TP1 每块 327,680 bytes。用运行时 GPU 进程映射传入 `--pid-gpu-map`。全部三组完成后用 [`plot_ratio_4x.py`](plot_ratio_4x.py) 生成共同时间轴的对照图。`run_manifest.json` 用 [`finalize_ratio_run.py`](finalize_ratio_run.py) 固化源码提交、配置、轨迹及原始文件哈希。

这是一轮单次压力回放，用于观察容量与有效产出随资源配比变化；无配对重复时不能给出统计显著性或自然流量发生率。该轨迹没有 urgent 元数据，原生 FCFS 无主动 keep，本轮也不能证明主动暂停或阶段缓冲的收益。
