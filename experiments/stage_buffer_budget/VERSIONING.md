# 固定工作树与实验版本

此研究的唯一日常代码入口是 `/home/zhongyu/project/vllm-omni-buffer`，Git 分支是 `exp/stage-buffer-inventory`。启动服务时，`--framework` 应指向该工作树。历史报告目录及按日期命名的源码副本仅用于核对旧运行，不能作为新运行的默认代码入口。

| commit | 内容 | 验证边界 |
| --- | --- | --- |
| `e7faad0c` | 2026-10-12 ServeGen active 冻结快照 | 与 `/home/zhongyu/project/vllm-omni-servegen-active-20261012` 的跟踪文件逐字节核对一致 |
| `4f5f3c86` | 阶段缓冲预算、100 ms 上限、控制器与配置 | 9 项 CPU 检查通过；该提交没有成功的 GPU 请求验证 |
| `5c055572` | 修复输出 CPU 张量视图在 IPC 中重复携带整个批次存储 | 9 项 IPC CPU 检查通过；修复的相同生产代码在先前独立副本中完成过真实 9 请求及 70 请求前后对照，本工作树未重新跑 GPU |

`exp/keep-execution-slots` 是执行名额 K 与驻留名额 R 分离的**独立实验分支**，当前没有合入。它允许暂停期间新请求准入，而现有缓冲策略会在暂停期间限制准入；二者不能在没有联合语义测试时直接合并。它的原始提交为 `93e362cf`，49 项 CPU 检查通过，并有此前两请求 GPU 正确性记录。

`fix/compact-cpu-tensor-storage` 和 `research/kv-residency` 是整理过程中生成的本地中间分支。日常修改、提交和服务启动均以本文件开头的固定工作树为准。先前未提交的 `vllm_omni/scheduling/request_queue.py` 删除保存在原仓库 stash 中，stash 对象号见报告目录 `version_management_20261015/before.json`；它没有混入缓冲策略分支。

运行前检查：

```bash
git -C /home/zhongyu/project/vllm-omni-buffer status --short
git -C /home/zhongyu/project/vllm-omni-buffer rev-parse --abbrev-ref HEAD
git -C /home/zhongyu/project/vllm-omni-buffer rev-parse HEAD
python3 /home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget/verify_budget_policy_cpu.py
```

运行清单需记录完整 commit、分支、dirty 状态、部署配置与负载 SHA-256、推理环境版本。FCFS/Ours 对照应固定同一个 commit，并仅切换明确的策略配置。旧报告保留原有源码路径和 hash，不追认成新 commit 的实测。

`/home/zhongyu/research_reports/kv_residency_trace_20261013/run_plateau_diagnostic.py` 是旧版诊断入口，默认仍指向诊断副本；它不能作为当前缓冲策略的正式实验入口。新的运行应使用本目录 README 所述的 `launch_server.py --framework /home/zhongyu/project/vllm-omni-buffer`。
