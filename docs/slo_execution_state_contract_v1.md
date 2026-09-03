# SLO 多阶段恢复：执行版本与输出提交契约 v1

> 阶段：P0.3a，设计冻结候选  
> 范围：只定义逻辑请求、流水线执行、阶段尝试和输出序号；不定义抢占策略  
> 基线：`ce213dad`，兼容完整流水线重启，并为 `aefeef00` 的阶段内重算预留统一语义

## 1. 设计目的

完整流水线重启和阶段内重算必须共用一套身份与状态规则，否则会出现：

- 旧执行的 late output 被当成新执行输出；
- 同一个物理 request ID 在阶段内重算后无法区分前后两次 attempt；
- 下游 stage 混用旧 bridge cursor 和新 replay 数据；
- 通过解析 `__executionN` 或 `__recoveryN__stageS` 字符串推断状态；
- 文本和音频输出使用一个总计数，导致跨流错误抑制；
- full restart 和 stage-local recovery 各维护一套互不兼容的字段。

本契约只解决正确性与可审计性，不承担性能收益主张，因此 P0 不需要论文调度基线。

## 2. 四层身份

### 2.1 Logical request

`logical_request_id` 是前端和实验统计唯一使用的请求身份，从 ingress 到最终完成保持不变。

### 2.2 Pipeline attempt

`pipeline_attempt_id` 表示从 Stage 0 开始的一次完整流水线执行：

- 初始执行为 0；
- 只有 `FULL_PIPELINE_RESTART` 才加 1；
- 阶段内重算不改变它；
- 它是显式整数，不能通过物理 request ID 字符串解析获得。

### 2.3 Stage attempt

`stage_attempt_id[s]` 表示阶段 `s` 当前有效的执行版本：

- 每个新 pipeline attempt 中，各阶段从 0 开始；
- 阶段 `s` 发生本地重算时，`s..final_stage` 的 stage attempt 分别加 1；
- `s` 之前的阶段 attempt 不变；
- 当前阶段即使复用相同的物理 request ID，也必须增加 stage attempt；
- 下游阶段无论复用或更换物理 ID，都必须增加 stage attempt。

有效阶段版本由三元组唯一标识：

```text
(pipeline_attempt_id, stage_id, stage_attempt_id)
```

### 2.4 Physical request

`physical_request_id` 只用于 engine、scheduler、connector 和资源 binding。它是 opaque string：

- 可以沿用当前 `logical__executionN`、`logical__recoveryN__stageS` 命名；
- 业务逻辑不得解析字符串获得 logical/pipeline/stage attempt；
- 显式 registry 保存 physical → logical 和 physical → stage version 映射；
- 清理逻辑必须覆盖该 logical request 历史上注册过的全部 physical ID。

## 3. 唯一事实源

Orchestrator 的 `OrchestratorRequestState` 持有唯一的 `ExecutionLedger`：

```python
@dataclass
class StageExecutionRef:
    logical_request_id: str
    pipeline_attempt_id: int
    stage_id: int
    stage_attempt_id: int
    physical_request_id: str
    started_monotonic_s: float


@dataclass
class ExecutionLedger:
    logical_request_id: str
    final_stage_id: int
    pipeline_attempt_id: int
    current_stages: dict[int, StageExecutionRef]
    physical_history: set[str]
```

`stage_request_ids`、`current_execution_id` 和 `_physical_to_logical_request_ids` 在迁移期可以保留为兼容视图，但只能由 ledger 更新，不能成为第二事实源。

Stage scheduler 不直接修改 ledger。它只接收当前 `StageExecutionRef`，在 output/preemption envelope 中原样返回对应的版本字段。

## 4. 输出提交状态

每个前端流独立维护 `OutputCommitState`：

```python
@dataclass
class OutputCommitState:
    stream_id: str                 # text / audio / future modality
    committed_sequence: int        # 已提交的最大逻辑序号，初始 -1
    replay_watermark: int          # 恢复开始时已经提交的最大序号
    terminal_committed: bool
```

每个模型输出 chunk 必须携带：

```text
logical_request_id
pipeline_attempt_id
stage_id
stage_attempt_id
stream_id
attempt_sequence
finished
```

`attempt_sequence` 是该阶段 attempt 从可重放边界开始产生的流内序号。若从阶段头重算，则从 0 重新开始；若从 checkpoint 恢复，则 checkpoint manifest 明确给出序号基点。

P0 移植阶段允许使用现有“按已输出 chunk 数抑制 replay”的实现作为兼容适配，但兼容层必须转换为 per-stream watermark，不能继续扩展 `final_outputs_emitted[stage_id]` 这种单计数设计。

## 5. 状态转换

### 5.1 初始提交

输入：logical request、final stage 和每阶段物理 ID。

结果：

- `pipeline_attempt_id = 0`；
- 所有 `stage_attempt_id = 0`；
- 注册 current stage ref 和 physical history；
- 所有 stream `committed_sequence = -1`；
- 不存在 replay watermark。

### 5.2 完整流水线重启

触发：任意阶段决定使用 `FULL_PIPELINE_RESTART`。

原子状态变化：

1. 校验 preemption envelope 的 stage ref 仍是 current；
2. `pipeline_attempt_id += 1`；
3. 旧 pipeline 所有 stage ref 退休；
4. 新 pipeline 所有 stage attempt 从 0 开始；
5. 清空所有运行时生成的 bridge/recurrent/connector cursor；
6. 为每个已提交前端流记录 replay watermark；
7. 新 Stage 0 从不可变输入提交；
8. 下游按新上游实际产生数据后因果启动；
9. 旧 physical ID 保留在 history 中，直到 logical request 清理。

### 5.3 阶段内重算

触发：阶段 `s` 决定使用 `STAGE_RECOMPUTE`。

原子状态变化：

1. 校验 stage `s` 的 preemption envelope 仍是 current；
2. pipeline attempt 不变；
3. stages `< s` 保持原 ref；
4. stage `s` 的 stage attempt 加 1，即使物理 ID 不变；
5. stages `> s` 的 stage attempt 加 1，并建立新的下游 consumer ref；
6. 只清理从 `s` 开始失效的动态状态和 connector cursor；
7. 为受影响的最终输出流记录 replay watermark；
8. abort/release 旧下游 physical binding；
9. 重放期间按 stream watermark 抑制重复输出。

### 5.4 正常阶段推进

当请求第一次进入尚未建立 ref 的下游阶段时，orchestrator 为该阶段建立当前 pipeline 下的 attempt 0。已有 ref 的阶段只允许 current physical/version 更新；旧版本数据不能覆盖 current ref。

### 5.5 完成与清理

logical request 只有在全部适用前端流 terminal committed 后完成。完成或 abort 时：

- abort/release physical history 中的全部 ID；
- 取消未完成 connector 和恢复事务；
- 失效全部 checkpoint/state manifest；
- 删除 physical registry、ledger 和 output commit state；
- 迟到消息因 registry/版本不存在而被丢弃并记录原因。

## 6. 消息接受规则

orchestrator 接收 stage output、preemption、connector ack 或 recovery completion 时，依次验证：

```text
1. logical request 是否仍存在
2. stage_id 是否在路径内
3. physical_request_id 是否注册给该 logical request
4. pipeline_attempt_id 是否等于当前 pipeline
5. stage_attempt_id 是否等于该 stage 当前 attempt
6. physical_request_id 是否等于该 stage 当前 physical ID
7. event monotonic timestamp 是否不早于 started_monotonic_s
```

任一条件失败均 fail closed，不修改状态，记录标准化 drop reason：

```text
unknown_logical_request
invalid_stage
unregistered_physical_request
retired_pipeline_attempt
retired_stage_attempt
retired_physical_request
event_before_attempt_start
duplicate_output_sequence
output_after_terminal
```

只有版本检查通过后，才做输出序号和 finished 处理。时间 cutoff 只能作为迁移期的额外保护，不能替代显式版本。

## 7. 输出提交规则

对版本有效的输出 `(stream_id, attempt_sequence)`：

1. 若 stream 已 terminal committed，丢弃为 `output_after_terminal`；
2. 若 `attempt_sequence <= replay_watermark`，丢弃为 replay duplicate；
3. 若 `attempt_sequence <= committed_sequence`，丢弃为普通 duplicate；
4. 若序号大于 `committed_sequence + 1`，暂不提交，记录 gap 并进入错误/等待策略；
5. 否则提交给前端并更新 `committed_sequence`；
6. `finished=True` 只有在之前不存在 gap 时才能提交 terminal；
7. replay 在 watermark 之前已经终止时，只允许发送 content-free terminal，并记录 `replay_ended_before_watermark`，供正确性分析识别。

文本和音频分别维护序号与 terminal，禁止用音频 replay 计数抑制文本，反之亦然。

## 8. 不变量

实现和测试必须持续验证：

- 每个 logical request 同时只有一个 current pipeline attempt；
- 每个路径内 stage 同时只有一个 current stage ref；
- pipeline/stage attempt 非负且只按定义转换；
- 任意 current physical ref 都能反查同一 logical request；
- physical history 只增不减，直到 logical completion/abort；
- full restart 不能保留旧 pipeline 的动态 bridge 状态；
- stage-local restart 不能清理上游有效状态，也不能保留下游失效状态；
- 已提交输出序号严格单调，terminal 每个 stream 最多一次；
- retired 消息不能改变 current ref、输出 watermark、提交序号或完成状态；
- 所有失败转换在返回时要么完全未发生，要么已经安全回退，不能停在半更新状态。

## 9. P0.3b 失败测试清单

下一小步先增加纯 CPU 测试，再接 orchestrator：

1. 初始 ledger 建立正确的 pipeline/stage attempt；
2. full restart 使 pipeline 加 1、stage attempts 重置，并退休全部旧 ref；
3. Stage 1 local restart 只更新 Stage 1/2；
4. Stage 2 local restart 不改 Stage 0/1；
5. 同 physical ID、旧 stage attempt 的 late output 被拒绝；
6. 新 physical ID、旧 pipeline attempt 的 preemption envelope 被拒绝；
7. text/audio watermark 相互独立；
8. duplicate、gap、terminal-after-terminal 被拒绝；
9. 失败转换不产生部分更新；
10. logical cleanup 返回完整 physical history。

通过这些纯状态测试后，P0.4 才把现有 full restart 字段迁入 ledger，并人工移植 stage-local recovery。

## 10. Trace 字段

所有恢复相关事件统一携带：

```text
logical_request_id
physical_request_id
pipeline_attempt_id
stage_id
stage_attempt_id
recovery_mode
transition_id
monotonic_s
```

拒绝事件额外记录 `drop_reason` 和消息携带的 observed version；状态转换事件记录 before/after version。后续实验只能使用这些显式字段关联事件，不再解析 request ID 名称。
