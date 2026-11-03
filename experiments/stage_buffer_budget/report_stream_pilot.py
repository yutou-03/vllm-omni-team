"""Build a transparent pilot report from preserved artifacts, without selecting seeds."""
import argparse,datetime,json,statistics
from pathlib import Path
from summarize_stream_pairs import summarize


def fmt(v,n=3):return '—' if v is None else f'{v:.{n}f}'


def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--final',action='store_true');a=p.parse_args();r=a.root;s=summarize(r);arms=s['arms']
 names={'native_k1':'裸原生 K=1','fixed':'固定阈值','dynamic':'动态阈值','matched':'同 K/R 无缓冲无暂停','paced':'动态暂停 + 两条边受控放行','buffer_only':'两条边缓冲放行，无暂停'}
 lines=['# 实验二：阶段缓冲与保留 KV 暂停——两小时试跑报告','',
  '状态：'+('本轮已收尾。' if a.final else '运行中草稿。')+' **机制验证和受控试跑；不属于正式性能结论。**',
  '', '## 1. 结果与解释','',
  '| 策略 | 成对种子数 | 流式分数 G | 相对裸原生 | 首输出 SLO 达标 | 完整响应 SLO 达标 |',
  '|---|---:|---:|---:|---:|---:|']
 for arm,v in arms.items():
  all_=v['groups']['all'];comp=s['comparisons'].get(arm+'_vs_native_k1');gain=fmt(comp['gain_percent'],2)+'%' if comp else '—'
  lines.append(f"| {names[arm]} | {len(v['seeds'])} | {fmt(v['G_stream_mean'])} | {gain} | {all_['first_slo_met']}/{all_['requests']} | {all_['e2e_slo_met']}/{all_['requests']} |")
 lines+=['','固定、动态、裸原生和同 K/R 诊断使用同一批五条到达轨迹。额外的受控放行消融如有结果，只使用预先指定的 seed 1103，不能与五组平均直接比较；其提升由该种子的同轨迹裸原生计算。',
  '', '| 比较 | G 相对变化 | 五种子配对 bootstrap 95% 区间 |','|---|---:|---:|']
 for key,v in s['comparisons'].items():
  ci=v['paired_bootstrap_95_percent_interval'];interval='单种子，不给置信区间' if ci is None else f'[{fmt(ci[0],2)}%, {fmt(ci[1],2)}%]'
  lines.append(f"| {key} | {fmt(v['gain_percent'],2)}% | {interval} |")
 lines+=['', '**约 20% 是用户期望的改进目标，不是必须得到的统计结论。**是否能称为实验二收益，还需同时看受害请求断流、SLO 和输出内容是否可比。固定/动态五组的统计范围只覆盖固定内容、不同到达抖动；该区间不能代表多样真实业务负载。',
  '', '## 2. 紧急请求与原请求的体验','',
  '| 策略 | 紧急文本 TTFT 均值 / P95(s) | 原宽松音频 TTFP P95(s) | 原音频 E2E P95(s) | 原音频累计断流均值(s) | 原音频最大单次断流 P95(s) |',
  '|---|---:|---:|---:|---:|---:|']
 for arm,v in arms.items():
  t=v['groups']['text_standard'];b=v['groups']['audio_loose']
  lines.append(f"| {names[arm]} | {fmt(t['mean_first_s'])} / {fmt(t['p95_first_s'])} | {fmt(b['p95_first_s'])} | {fmt(b['p95_e2e_s'])} | {fmt(b['mean_rebuffer_s'])} | {fmt(b['p95_max_rebuffer_s'])} |")
 lines+=['', '“原请求”在这里指每周期先运行、允许成为暂停候选的宽松长音频请求。完整分层数据（标准/宽松、文本/音频、每个请求）在 `paired_summary.json` 和各次运行 `stream_score.json`。如果音频断流上升，即使总分增加也不能声称原请求体验无损。',
  '', '### 输出内容可比性','', '| 策略 | 每条宽松音频平均内容秒数 | 每条标准音频平均内容秒数 |','|---|---:|---:|']
 for arm,v in arms.items():
  lines.append(f"| {names[arm]} | {fmt(v['groups']['audio_loose']['mean_content_s'])} | {fmt(v['groups']['audio_standard']['mean_content_s'])} |")
 lines+=['','跨服务进程的生成输出有差异，不能把所有音频尾延迟变化直接解释为调度加速。暂停正确性由**同一个存活服务进程内**的暂停/无暂停参考逐 token 和逐 WAV 字节比较验证；该验证不能消除跨策略重启产生的内容差异。主分数仍按冻结公式如实计算，不能事后按内容长度重加权。',
  '', '## 3. 暂停事件链、库存与实际墙钟','',
  '| 策略 | 申请 / 完整链 | 实际恢复执行 P95 / 最大(ms) | 释放超期最大(ms) | 恢复执行超期最大(ms) | 恢复执行后新 chunk P95(ms) |',
  '|---|---:|---:|---:|---:|---:|']
 for arm in arms:
  f=r/(arm+'_pilot_audit.json')
  if not f.exists():continue
  audit=json.load(f.open());c=audit['chain_summary']
  lines.append(f"| {names[arm]} | {c['requested']} / {c['complete']} | {fmt(c['actual_pause_ms_p95'])} / {fmt(c['actual_pause_ms_max'])} | {fmt(c['release_overrun_ms_max'])} | {fmt(c['resume_overrun_ms_max'])} | {fmt(c['resume_to_new_chunk_ms_p95'])} |")
  if audit['inventory']['conservation_violations'] or audit['chains']['integrity_errors']:lines.append('**该策略存在完整性审计错误，不能报告为机制通过。**')
 lines+=['','完整链为 `small_hold_requested → urgent_keep_admitted + stage_first_forward → small_hold_released → small_hold_first_rescheduled → 新发布 chunk`。KV 块保留在 GPU，紧急请求真实进入 K=1 批次，暂停者在此期间不执行。',
  '', '**100 ms 只是申请上限。**标志释放和原请求真正重新执行是两件事；紧急请求可能继续占有执行名额。因此本轮控制器额外要求接收端库存覆盖紧急请求完整预计服务时间、恢复余量和边界开销。报告分别列出两类超期，不把标志释放当作已经恢复。',
  '', '库存按已接收 conditioning 位置或新 codec 帧计算，排除 prompt、重复 codec left context 和上游尚未被接收的数据。接收、释放游标独立；FIFO 直接持有 get 后反序列化对象的引用。每请求上限 128 chunk、1024 单位、64 MiB，全部请求 512 MiB。队列满停止预取；当前实现还没有全链路限制上游共享内存发布的背压。',
  '', '### 机制烟测证据','']
 for arm in ['stage0','fixed','dynamic','stage1','paced']:
  ap=r/(arm+'_mechanism_audit.json');pp=r/('mechanism_output_proof.json' if arm=='stage0' else arm+'_output_proof.json')
  if not ap.exists():continue
  v=json.load(ap.open());proof=json.load(pp.open()) if pp.exists() else None
  lines.append(f"- {arm}: 完整链 {v.get('complete_chains')}; 接收队列入/出 {v.get('enqueued_chunks')}/{v.get('released_chunks')}; 完整性错误 {v.get('integrity_errors')}; 输出证明文件 `{pp.name}` {'已保存' if proof else '尚无'}。")
 lines+=['', 'Stage 0 和 Stage 1 的机制分开验证。本轮没有实现同一请求双阶段暂停的全局协调，也没有把 Stage 1 暂停收益写入 Stage 0 的五组成对结果。文本客户端反馈保护的暂停分支亦未作为本轮性能策略。',
  '', '## 4. 冻结条件与评分','',
  '- 模型：Qwen3-Omni-30B-A3B-Instruct；物理 GPU 0–3 为 Stage 0 TP4，GPU 4 为 Stage 1，GPU 5 为 Stage 2，均为 A100 40GB。',
  '- 五条冻结轨迹：1103、2207、3301、4409、5519；每条 12 请求、60 秒到达窗口、平均 0.2 QPS；文本/音频各 50%，标准/宽松各 50%。所有策略使用相同逐条到达、输入、生成 seed 和 SLO 元数据。',
  '- 标准文本 TTFT 1.3 s/E2E 3 s，宽松文本 2.6 s/6 s；标准音频 TTFP 2.5 s/E2E 45 s，宽松音频 3 s/120 s。独立标定只有每层 3 个样本，故不能称充分校准。',
  '- Thinker 温度 0、seed 42；Talker 保留部署采样参数。实际源码快照、容器镜像 ID、模型配置指纹、负载和参数哈希都保存。',
  '- 各比较组 Stage 0 执行名额 K=1。裸原生 max_num_seqs=1，暂停组原生驻留配置 R=64；物理 KV 池字节相同（Stage 0 每 rank 14,136,115,200，Stage 1 1,894,318,080）。所以另测相同 K=1/R=64 的无缓冲无暂停诊断，不把结构差异隐去。这个实验不是默认 K=64 的生产容量提升。',
  '- 不改变原生 FCFS 次序；不采用实验一余量排序，不换出、不重算 KV。',
  '', '可复制公式：','', '```latex',
  r'G_{\mathrm{stream}}=\frac{1}{T_{\mathrm{arrival}}}\sum_{i\in\mathcal A}',
  r'\mathbf{1}\!\left\{C_i=1,\;\mathrm{TTF}_i\le F_i\right\}',
  r'\left[\sum_j m_{i,j}\,h_{i,j}\,w(B_{i,j})-\mu R_i\right]_+',
  '```', '',
  '文本每个客户端可见 tokenizer 单位 m=1，消费 4.8 token/s；音频按真实 1× 播放，拆成至多 20 ms 单位，m=3.3×内容秒数。B 是客户端收到但未消费的覆盖秒数，不是阶段队列。h=max(0,1−迟到秒数/1s)，w=max(0,1−max(B−10s,0)/20s)，μ=3.3。理想消费时间线从实际首输出开始，R 是累计播放/阅读缺货时间。',
  '', '完整输出和首输出 SLO 决定门槛；E2E、最大断流另作体验约束。分母固定为到达窗口，全部 offered 请求进入统计，窗口后继续排空再评分。它是固定到达队列的流式服务分数，不是仅凭该数就能推断的最大 QPS。',
  '', '## 5. “超过 512 token 后预测更准”的验证','']
 pred=r/'prefill_prediction/summary.json'
 if pred.exists():
  v=json.load(pred.open());lines+=['原导图将该假设关联到 prefill 近线性，因此这里按**实际 prompt token 数**分组，测 Stage 0 首次 forward 至首输出的服务时间；不把输出长度错当成 prefill 长度。训练和验证重复分开，启动样本排除。',
   '', '| 验证分组 | 样本数 | MAPE | MAE(ms) | P95 APE |','|---|---:|---:|---:|---:|']
  for name,g in v['groups'].items():lines.append(f"| {name} | {g['n']} | {fmt(g['MAPE_percent'],2)}% | {fmt(g['MAE_ms'])} | {fmt(g['p95_APE_percent'],2)}% |")
  lines+=['','这里只能验证单一文本 prompt 家族、独立请求下的预测误差；不能证明繁忙同步批次的 checkpoint 等待时间同样准确，也不能证明 512 是通用拐点。包含输出处理开销，不能称纯 GPU kernel 时长。']
 else:lines.append('本轮尚未形成该验证结果；不依据 64/256 输出 token 的暂停试跑推断该假设成立。')
 diagnosis=r/'gate_delay_diagnosis.json'
 if diagnosis.exists():
  d=json.load(diagnosis.open());lines+=['', '## 6. 已定位问题与下一版验证', '',
   f"15 个宽松音频请求中，动态组首两段音频间隔相对原生的中位增量为 {d['median_extra_first_audio_gap_s']:.3f} s，与固定 0.5 s 蓄水闸门相符。部分闸门在 FIFO 已有 77/82 个单位、暂停已经开始后仍阻塞下游。这是观测定位，尚需单独消融验证因果。",
   '', '下一版在 FIFO 达到 64 个已接收单位后结束/跳过蓄水等待；Stage 2 以 16 新 codec 帧/s 为基础放行，并按已接收积压/剩余完成期限加速排空，预留 5 秒。未降低原暂停安全阈值。可变放行速率不再为 Stage 1 暂停提供固定速率保证；扩展性能只暂停 Stage 0。',
   '', '该版本在看到原五组结果后提出，独立标记为探索。扩展接收队列 9 项 CPU 测试、控制器 23 项通过。为遵守两小时限制，本轮没有启动其 GPU 性能重放，因此不能把预期改善记为已获得收益。源码快照、配置和单种子重放计划已保存到 paced_framework_v2、receiver_paced.json 和 paced_exploration_plan.json。',
   '', 'Stage 1 烟测实际恢复执行后至新 chunk 为约 322 ms，而当前基于调度迭代的恢复保护量约 36 ms。此次已有 6.25 s 保守库存覆盖，实际完整恢复链约 1.08 s，没有断流；但正式 Stage 1 性能比较前，仍须把“拼成下一个可发布 chunk”的成本单独校准，不能仅用一次迭代时间代替。']

 lines+=['', '## 7. 复现与下一步','',
  '运行入口在仓库 `experiments/stage_buffer_budget/`：`launch_exp2_container.py` 冻结服务源码及部署；`run_frozen_stream.py` 开放环重放并排空；`stream_goodput.py` 按冻结指标重算；`audit_pilot_streams.py` 审计真实请求；`summarize_stream_pairs.py` 成对统计。原始 SSE、WAV、阶段事件及每次请求结果均保留。',
  '', '```bash',
  f'R={r}',
  'E=/home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget',
  'python3 "$E/summarize_stream_pairs.py" "$R"',
  'python3 "$E/audit_pilot_streams.py" --root "$R" --arm dynamic',
  '# stream_goodput.py 需要镜像内的 transformers；以已有服务容器为例：',
  'docker exec <container> python "$E/stream_goodput.py" --workload "$R/pilot_s1103.jsonl" --run "$R/dynamic_pilot_s1103" --config "$R/metric_pilot.json"',
  '```','',
  '下一步优先验证客户端已接收播放库存与阶段库存的联合保护，校准同步批次和恢复首 chunk 的保守上界，并在完全相同 K/R 和生成执行配置下减少跨进程输出差异。再扩展内容、随机运行顺序及负载点，完成双阶段协调；不要仅提高触发次数或放宽 SLO 来追求 20%。',
  '', 'CPU 验证：接收缓冲与暂停原语 56 项通过；控制器 22 项通过，日志位于 `receiver_and_hold_tests.log`、`controller_tests.log`。此前适配器测试中的 cleanup 失败在未修改基线也复现，不当作本轮新功能通过的证据。',
  '', '关键索引：`paired_summary.json`、`pilot_workload_manifest.json`、`metric_pilot.json`、各 `*_pilot_audit.json`、`*_output_proof.json`、`runtime_fingerprint.json`、每个 `*_service/manifest.json`、`source_changes.patch`、`artifact_manifest.json`。',
  '', '生成时间：'+datetime.datetime.now(datetime.timezone.utc).isoformat()]
 (r/'RESULTS_20261102.md').write_text('\n'.join(lines)+'\n')
 # Standard exportable plot: primary five-seed comparison, with audio cost visible.
 try:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  order=[x for x in ['native_k1','fixed','dynamic','matched'] if x in arms];labels=['Native','Fixed','Dynamic','Matched nohold'][:len(order)]
  fig,axes=plt.subplots(1,3,figsize=(13,4));colors=['#64748b','#d97706','#2563eb','#059669'][:len(order)]
  metrics=[('Streaming score','G_stream_mean'),('Urgent text mean TTFT (s)','ttft'),('Victim mean total stall (s)','stall')]
  for ax,(title,k) in zip(axes,metrics):
   y=[arms[x]['G_stream_mean'] if k=='G_stream_mean' else arms[x]['groups']['text_standard']['mean_first_s'] if k=='ttft' else arms[x]['groups']['audio_loose']['mean_rebuffer_s'] for x in order]
   bars=ax.bar(labels,y,color=colors)
   ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_title(title);ax.tick_params(axis='x',rotation=15);ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(y)*1.2)
   if k=='ttft':ax.axhline(1.3,color='#dc2626',ls='--',lw=1,label='1.3 s SLO');ax.legend(fontsize=8)
  fig.suptitle('Experiment 2: controlled K=1 pilot — five paired arrival seeds, fixed score',fontsize=12);fig.tight_layout();fig.savefig(r/'paired_stream_metrics.png',dpi=180);plt.close(fig)
 except ImportError:pass
 print(str(r/'RESULTS_20261102.md'))
if __name__=='__main__':main()
