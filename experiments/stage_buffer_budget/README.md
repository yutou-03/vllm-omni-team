# 阶段库存预算策略：隔离的后续版本

历史 `servegen_active_buffer_20261012` 运行及其源码清单保持原样。新框架在 `/home/zhongyu/project/vllm-omni-buffer`，新策略入口是仓库根目录的 `stage_buffer_controller_budget:decide`；启动环境由 `experiments/stage_buffer_budget/env.json` 指定，控制器默认读取同目录的 `controller_budget_config.json`。此长期工作树使用 Git 分支 `exp/stage-buffer-inventory`，历史实验快照只用于复现。

## 到期延后的处理

旧运行在 Stage 0 暂停后的下一轮实际调度了另一个请求的 13,180-token prefill，约 4 秒后才再次进入 Stage 0 调度。日志支持大型批次占住同步执行路径这一解释，但没有单独证明 4 秒全部由该 prefill 计算造成。新版本在策略暂停期间推迟等待队列的接纳，将单轮 token 预算限制为最多 64；申请时和每轮调度前都检查其他驻留请求是否仍处于解码边界，没有解码同伴或同伴转入非解码工作则立即结束暂停。推迟等待队列的接纳会影响同阶段其他请求的首 token 时延，收益需要在正式运行中测量。若下一轮检查仍比截止时间晚超过 50 ms，记录 `stage_buffer_policy_suspended` 并停止该实例后续主动暂停。

同步 `EngineCore.step()` 等待模型执行结束，无法中断已经启动的 GPU 批次。因此这些改动消除了已定位的大型 prefill 路径并提供超期停用保护，**仍不构成 100 ms 实际暂停时长的严格保证**。严格墙钟上界需要能在执行中途安全唤醒/中断或有经过证明的执行切片上界；当前同步运行栈没有这一能力。

## 申请时长

按请求统计下游已接收且尚未取用的 `Q_ready`，并与上游已发布库存取较小值。消费速率上界沿用参考标定下限与最近最多 64 个在线样本的 P95 最大值。令 `H=Q/c_upper`，`low=max(resume_below_s, scheduler_tick_p95+margin_s)`，本次请求时长为 `min(100 ms, max(0, H-low-margin_s))`。预算不足 10 ms 时不暂停；暂停中库存下降或期限到达仍会请求恢复。原始发布库存和接收端库存都会写入决策事件。

`controller_budget_config.json` 的阈值是初始实验值，不代表已校准的恢复延迟保证。Stage 0 的基础低水位为 150 ms、高水位为 170 ms；Stage 1 分别为 30 ms 和 50 ms。策略继续保留请求 KV 和驻留名额，不会凭暂停增加可接纳请求数。

## 检查与启动

CPU 检查：`python3 experiments/stage_buffer_budget/verify_budget_policy_cpu.py`。历史 `mixed_high` 事件的只读回放验证了 53 个请求的发布、接收和取用计数最终守恒，未发现负库存。

需要实际运行时，可用现有 `launch_server.py`，传入 `--framework /home/zhongyu/project/vllm-omni-buffer` 和 `--extra-env-file /home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget/env.json`，并使用新的 session 名称。2026-10-14 的首次 canary 在模型加载前因历史运行环境缺少 `transformers` 而退出，未发送推理请求；服务已停止。服务启动、实际超期分布、输出正确性和性能收益尚待 GPU 验证。

## 2026-10-15 申请上限试验

根据用户要求，新策略将每次申请上限从原 50 ms 放宽至 100 ms；库存预算仍可申请更短时间，预算不足 10 ms 不暂停。原语硬上限同步改为 100 ms。上一次误将上限收紧至 20 ms 的中间改动已撤回；超期停用阈值恢复为晚到 50 ms。申请上限与检查是否晚到是两个独立参数。放宽上限不能中断正在运行的 GPU 批次，无法单独解决曾出现的约 4 秒调度空窗。旧 `mixed_high` 的 Stage 0 申请中位数 50 ms，实际释放中位数约 58.3 ms、P95 约 82.1 ms、最大约 4 秒；100 ms 是对照实验起点，尚无新策略 GPU 实测。
