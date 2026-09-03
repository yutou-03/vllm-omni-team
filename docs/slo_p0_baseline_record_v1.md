# SLO 多阶段调度 P0 基线记录 v1

## 范围

本记录只验证后续集成所依赖的调度元数据、队列策略、调度器 mixin、chunk coordinator 和完整流水线重启单元测试。它不是 GPU 端到端性能基线，也不证明新策略收益。

## 代码基线

- 集成分支：`exp/slo-aware-multistage-v1`
- 新 worktree：`/home/zhongyu/project/vllm-omni-slo-v1`
- 功能基线：`ce213dad33acb8a303223cd133bc1d39fb5b4903`
- 基线来源分支：`exp/full-pipeline-restart`
- 测试契约修复：`855d395a`

该 worktree 从 `ce213dad` 直接创建。没有合入原有 dirty worktree 的代码、日志或实验结果。

## 测试环境

- 日期：2026-09-03
- 容器：`vllm_omni_privileged`
- 镜像标签：`vllm/vllm-openai:v0.20.0`
- 镜像 ID：`sha256:7f435fa6e65079f5b15bcefbee8d02220741c6857031f11292711e2ec07ca56a`
- vLLM-Omni：`0.20.0.dev0`
- 容器内 vLLM：`0.19.2rc1.dev219+gc0879d948.d20260618`
- GPU：8 × NVIDIA A100-PCIE-40GB
- 驱动：595.58.03

环境存在 vLLM-Omni 0.20 与 vLLM 0.19.2 的版本告警。当前选择的测试全部通过，但执行 GPU 端到端基线前必须单独确认该容器是否就是历史实验冻结环境；不能把本次单测通过解释为版本完全兼容。

## 测试命令

```bash
docker exec vllm_omni_privileged bash -lc '
  cd /home/zhongyu/project/vllm-omni-slo-v1 &&
  python3 -m pytest -q \
    tests/scheduling \
    tests/engine/test_scheduling_metadata_propagation.py \
    tests/engine/test_full_pipeline_preemption_recovery.py \
    tests/core/sched/test_omni_scheduler_mixin.py \
    tests/core/sched/test_chunk_scheduling_coordinator.py
'
```

## 初始结果与诊断

第一次运行结果：`90 passed, 3 failed`。

失败均来自基线已有测试与已有实现不同步，不是新运行时代码导致：

1. `test_chunk_policy_metadata.py` 使用 `__new__` 绕过构造函数，但没有建立 `OmniTransferAdapterBase` 后来增加的 `_drain_state_lock` 和 `_cancelled_load_reqs`。
2. 同一用例的 AR/generation 两个参数因此分别失败。
3. `test_chunk_scheduling_coordinator.py` 仍期待旧的 `{"left_context_size": 25}`，而实现自 `239035053` 起已标准化为 `{"meta": {"left_context_size": 25}}`。

处理原则：只更新测试 fixture 和断言以匹配现有运行时契约，没有修改运行时实现。

## 修复后结果

```text
93 passed, 17 warnings in 15.82s
```

单独重跑原失败文件：

```text
29 passed, 17 warnings in 5.44s
```

## P0.2 结论

- 选定的 CPU/逻辑层基线测试通过。
- `ce213dad` 可作为统一恢复状态设计的起点。
- 测试契约修复已作为独立 commit 保存，没有夹带运行时变更。
- 进入 P0.3 前不再扩大测试范围；GPU 端到端零行为变化验证留到 P0.5，在总开关和统一状态移植完成后做配对测试。

## 下一步

P0.3 只设计并测试统一的 logical request、pipeline execution、stage attempt 和 output sequence 契约。先补失败用例和状态转换表，再修改恢复实现。
