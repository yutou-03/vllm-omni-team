"""Generate the revision report without substituting old K1 or hard-gate scores."""
import argparse,json,datetime
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root;d=json.loads((r/'multibatch_summary.json').read_text());old=json.loads((r/'legacy_k1_v2_summary.json').read_text())
 names={'native64':'原生默认并发上限64','nohold16':'K16/R64 无新增缓冲无暂停','dynamic16':'K16/R64 成本感知暂停','fixed16':'K16/R64 固定预算暂停','nohold8':'K8/R64 无新增缓冲无暂停','dynamic8':'K8/R64 成本感知暂停','guard16':'K16/R64 400ms首输出保护（探索）'}
 lines=['# 实验二修订：多请求并行与连续首输出违约扣分','', '## 1. 本轮修订与结论范围','', '本轮7组各3条轨迹，共1260个成对请求及60个平滑参考请求已完成。β=3.3下，主动态K16相对同配置无暂停为−0.22%；K8敏感性为−2.08%。约20%的提升未实现。400ms保护探索版本的账面+1.07%伴随36/36音频样本数和样本哈希不一致及断流增加，不能当作可比性能收益。','', '旧 K=1 是机制隔离配置，会放大长请求阻塞紧急请求。原先 +12.95% 使用“完整交付且首输出达标才记分”的旧公式，不能用于新公式或正常多请求并行场景。','', '本轮保留原生默认 max_num_seqs=64 参考，主成对比较 K=16/R=64，另测 K=8/R=64 敏感性。K 是本阶段执行名额，R 是驻留请求上限；K=16/8 仍是受控配置，并非默认生产配置。没有把降低并发造成的损失当作原生系统必然存在的问题。','', '## 2. 新公式与冻结参数','', '```latex',r'G_{\mathrm{stream}}=\frac{1}{T}\sum_{i\in\mathcal A}\left[\sum_j m_{i,j}h_{i,j}w(B_{i,j})-\beta V_i-\mu R_i\right]_+',r'V_i=\max(0,\mathrm{TTF}_i-F_i)','```','', '文本 TTF 为 TTFT；音频为首段可播放时间 TTFP。删除完整性和首输出达标的指示函数，已收到的部分有效内容也按同一规则计分；完整率、首输出与 E2E SLO、断流另列。没有任何有效内容时贡献为0。','', 'β=3.3 等效内容单位/秒，在新成对 GPU 结果出现之前固定；敏感性为1.65、3.3、6.6、13.2。μ=3.3、文本4.8 token/s、音频1×播放并按3.3单位/内容秒换算、音频≤20ms拆分、迟到衰减1s、库存满分10s/衰减20s，均保持上一轮设定。T=75s到达窗口，所有到达请求纳入并排空后评分，不能仅由这个分数推断最大稳定QPS。','', '## 3. 同一批旧 K1 日志重算（不是新 GPU 结果）','', '| β | 原生K1 | 动态K1 | 动态相对提升 |','|---:|---:|---:|---:|']
 for b,v in old['results'].items():lines.append(f"| {b} | {v['native_k1']:.4f} | {v['dynamic']:.4f} | {v['dynamic_gain_percent']:+.2f}% |")
 lines+=['','该变化揭示旧结论对硬门槛的敏感性；不能挑选β使收益达到目标。','', '## 4. 新 GPU 重测','', '| 策略 | 种子数 | 请求数/完成 | G均值 | 首输出达标 | E2E达标 | 真实暂停申请/完整链 | Stage0观察到的最大首入批大小 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for arm,v in d['arms'].items():
  g=v['groups']['all'];m=v.get('mechanism',{});lines.append(f"| {names[arm]} | {len(v['runs'])} | {g['requests']}/{g['complete']} | {v['G_mean']:.4f} | {g['first_slo_met']}/{g['requests']} | {g['e2e_slo_met']}/{g['requests']} | {m.get('requested',0)}/{m.get('complete_chains',0)} | {m.get('actual_first_forward_peak_batch',{}).get('0','—')} |")
 lines+=['', '| 比较 | 新公式G变化 | 配对种子 |','|---|---:|---|']
 for label,v in d['comparisons'].items():lines.append(f"| {label} | {v['gain_percent']:+.2f}% | {v['seeds']} |")
 lines+=['', '**没有真实暂停的组，分数差异不能解释为暂停策略收益。**新增缓冲也可能改变首段时机、后续供给和音频内容；必须结合完整事件链及输出可比性解释。','', '### 请求体验','', '| 策略 | 紧急文本TTFT均值/P95(s) | 紧急文本首输出达标 | 宽松音频累计断流均值(s) | 音频最大单次断流P95(s) | 音频内容均值(s) |','|---|---:|---:|---:|---:|---:|']
 for arm,v in d['arms'].items():
  t=v['groups']['text_standard'];q=v['groups']['audio_loose'];lines.append(f"| {names[arm]} | {t['first_mean_s']:.3f}/{t['first_p95_s']:.3f} | {t['first_slo_met']}/{t['requests']} | {q['mean_rebuffer_s']:.3f} | {q['max_rebuffer_p95_s']:.3f} | {q['mean_content_s']:.3f} |")
 lines+=['', '### β 敏感性','', '| 比较 | β=1.65 | β=3.3 | β=6.6 | β=13.2 |','|---|---:|---:|---:|---:|']
 for label,v in d['comparisons'].items():lines.append('| '+label+' | '+' | '.join(f"{v['beta_gain_percent'][str(b)]:+.2f}%" for b in [1.65,3.3,6.6,13.2])+' |')
 lines+=['', '## 5. 配置和可比性','', '- 模型Qwen3-Omni-30B-A3B-Instruct，A100 40GB；物理GPU 1/2/3/5运行Stage0 TP4，GPU6运行Stage1，GPU7运行Stage2。GPU0/4当时被其他任务使用，未占用或停止其任务。所有新组映射一致，但与旧K1不是同一次硬件运行，因此旧K1只单独重算、不跨负载比较G绝对值。','- 固定物理KV池：Stage0每rank14,136,115,200 bytes；Stage1为1,894,318,080 bytes。K16和K8各自比较组R=64一致；原生默认参考不启用执行/驻留拆分。','- 每种策略三条共同轨迹，每条60请求/75s，平均0.8 QPS；80%文本/20%音频，80%宽松/20%紧急。每25秒一轮温和受控突发，16个宽松请求分布于约2.7秒，其后4个紧急文本；另有原生平滑到达参考。合成自然语言主题，关闭ignore_eos，文本输出上限64/128/256/512，音频上限256。输入和到达在看策略结果前冻结。','- 标准文本TTFT1.3s，宽松文本5.2s，宽松音频TTFP6s。E2E采用独立原生标定服务时长的倍数和类别下限；宽松文本≥60s、宽松音频≥120s。完整逐请求SLO在冻结jsonl内。标定每层只有两个暖态样本，不能当作充分校准。','- guard16 是在主动态组结果出现后提出的独立探索：仅把首输出保护预算从200ms改为400ms，取原生参考最大紧急TTFT 385.62ms向上取整到50ms；不改变安全库存阈值、负载、SLO、β或完整紧急服务预算。它没有统一客户端与服务端的计时起点，只增加了基于参考观测的前置开销保护。', '- 固定组与动态组共用在线消费速率、checkpoint边界检查、恢复保护和基础阈值。固定完整紧急服务预算5.52s；动态按请求类别使用1.26/1.91/5.52s。它们来自独立标定并在成对运行前冻结，尚不是准确覆盖所有批次大小的预测器。','- 接收队列在64个已接收单位时结束/跳过0.5s蓄水等待，两种暂停策略完全相同；未启用上轮未验证的Stage2可变放行时钟。只暂停Stage0，Stage1全程保持原生执行；不混入实验一余量排序，不换出或重算KV。','- 原生与各策略分组顺序执行，未做随机交错重复；存在时序和生成内容变化的混杂。默认64与K16差异同时包含并发上限变化，不能把两者差值全部归因于暂停。','', '## 6. 审计与复现','', '各 *_mechanism_audit.json 保存真实批次、申请/入批/释放/重新执行/新chunk链、接收队列顺序及阶段发布/接收计数。报告完整链时必须同时核对库存安全与客户端断流，100ms只是暂停申请上限。','', '运行入口：launch_exp2_container.py（新增 --gpus）、run_frozen_stream.py、stream_goodput.py（metric_version=soft_ttft_v2）、summarize_multibatch_v2.py、report_multibatch_v2.py。配置、输入哈希、容器命令、源码快照、原始SSE/WAV与阶段日志均在本目录。旧日志重算在legacy_k1_rescore_v2，未覆盖旧结果。','', '```bash', 'R='+str(r),'E=/home/zhongyu/project/vllm-omni-buffer/experiments/stage_buffer_budget','python3 "$E/verify_stream_metric_v2.py"','python3 "$E/summarize_multibatch_v2.py" "$R"','python3 "$E/report_multibatch_v2.py" "$R"', '# 使用冻结模型配置、源码和输入重新运行GPU实验（输出目录必须不存在）：', 'python3 "$E/replay_multibatch_v2.py" --frozen-root "$R" --output-root /home/zhongyu/research_reports/stage_buffer_experiment_20261102/replay_v2_new --gpus 1,2,3,5,6,7','```','', '生成时间：'+datetime.datetime.now(datetime.timezone.utc).isoformat()]
 extra=['## 7. 主比较的收益分解与限制','']
 if (r/'score_decomposition.json').exists():
  dec=json.loads((r/'score_decomposition.json').read_text());extra+=['| K16组 | 内容效用/T | 首输出违约扣分/T | 断流扣分/T | 最终G |','|---|---:|---:|---:|---:|']
  for name in ['nohold16','dynamic16','fixed16']:
   v=dec[name];extra.append(f"| {name} | {v['utility']:.4f} | {v['ttf_penalty']:.4f} | {v['rebuffer_penalty']:.4f} | {v['score']:.4f} |")
  extra+=['', '在保持内容效用项不变的条件下，K16无暂停组即使完全消除两项显式扣分，分数最多增加约1.59%。这不是整个策略空间的上限：改善及时交付权重h和积压权重w仍可增加内容效用；但不能继续把“救回几个首输出超时请求”预期为20%的收益。','']
 if (r/'dynamic16_timing_diagnosis.json').exists():
  t=json.loads((r/'dynamic16_timing_diagnosis.json').read_text());extra += [f"15次主动态组暂停中，服务端口径已超首输出期限{t['already_late_at_hold_server_clock']}次，客户端计划到达口径已超期{t['already_late_at_hold_client_clock']}次。控制器剩余预算比客户端口径乐观的中位差为{t['optimistic_budget_gap_ms_median']:.1f}ms，最大{t['optimistic_budget_gap_ms_max']:.1f}ms。入批后首输出真正超过200ms的只有{t['admission_to_first_exceeds_200ms']}次。不能把全部超时归咎于模型首输出计算。",'']
 if (r/'main_output_comparability.json').exists():
  cc=json.loads((r/'main_output_comparability.json').read_text());extra+=['| 相对nohold16 | 文本token序列相同 | 音频逐chunk样本哈希相同 |','|---|---:|---:|']
  for name,v in cc.items():extra.append(f"| {name} | {v['text_ids_identical_to_nohold16']}/{v['requests']} | {v['audio_chunk_samples_identical_to_nohold16']}/{v['audio_requests']} |")
  extra+=['', '主比较存在少量跨运行音频差异；guard16的36/36音频样本数及样本哈希均与对照不一致，因此其+1.07%不能作为可比性能收益。另有每个运行自身生成端到客户端的token计数、逐音频chunk样本长度和阶段FIFO顺序审计，结果见 *_delivery_audit.json。','']
 extra+=['验证记录：新评分公式8项针对性检查通过；暂停原语49项CPU检查通过。第一次CPU测试受缺失pytest全局插件配置阻止，改用独立配置后49项通过。新的一键重放包装器已通过语法与CLI检查，未在本轮再次完整执行；本轮GPU结果来自归档的实际运行驱动。','', '当前Stage0暂停候选依赖Thinker→Talker接收库存，宽松纯文本请求的客户端库存尚未用于暂停；不能称为覆盖所有路径的完整消费感知调度。下一步应统一预算起点、校准批次相关首输出与恢复成本，并单独验证文本客户端库存以及多个紧急请求同时等待时的名额分配。','', '![多并发新指标结果](multibatch_v2_metrics.png)','']
 lines.extend(extra)
 (r/'RESULTS_V2.md').write_text('\n'.join(lines)+'\n');print(r/'RESULTS_V2.md')

if __name__=='__main__':main()
