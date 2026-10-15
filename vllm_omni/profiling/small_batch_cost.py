"""Opt-in startup cost screen on real Stage1 weights, synthetic fixed KV.

No live requests exist here. This is a compute-opportunity screen, not an
end-to-end speedup measurement or a replay of a natural-language prefix.
"""
import json
import os
from pathlib import Path
import statistics
import time

import torch


@torch.inference_mode()
def startup_cost_screen(runner):
    target = os.environ.get("VLLM_OMNI_SMALL_BATCH_COST_FILE")
    if not target or int(getattr(runner.model_config, "stage_id", -1)) != 1:
        return
    if not runner.has_talker_mtp or getattr(runner.model, "talker", None) is None:
        raise RuntimeError("cost screen requires the complete Talker plus MTP path")
    if os.environ.get("SMALL_COST_CALIBRATE_S16") == "1":
        return calibrate_s16(runner, target)
    # Allocated caches are not yet assigned to any request. Use repeatable
    # synthetic history instead of uninitialized allocator contents.
    for cache in runner.kv_caches:
        if isinstance(cache, torch.Tensor):
            cache.zero_()
    for name in ("input_ids", "inputs_embeds", "talker_mtp_input_ids",
                 "talker_mtp_inputs_embeds", "last_talker_hidden", "text_step"):
        buffer = getattr(runner, name, None)
        if buffer is not None:
            buffer.gpu[:2].zero_()
    runner.positions[:2].fill_(511)
    if runner.uses_mrope:
        runner.mrope_positions.gpu[:, :2].fill_(511)
    records = []
    def forward(n):
        return runner._dummy_run(n, uniform_decode=True, force_attention=True,
                                 allow_microbatching=False, profile_seq_lens=512)
    for n in (1, 2):
        for _ in range(8):
            forward(n)
    torch.cuda.synchronize()
    for pair in range(20):
        for n in ((1, 2) if pair % 2 == 0 else (2, 1)):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before = time.perf_counter()
            start.record()
            output = forward(n)
            end.record()
            end.synchronize()
            elapsed = (time.perf_counter() - before) * 1000
            records.append(dict(pair=pair, batch=n, wall_ms=elapsed, gpu_interval_ms=start.elapsed_time(end),
                                output_finite=all(bool(torch.isfinite(t).all()) for t in output)))
    summary = {str(n): {k: statistics.median(r[k] for r in records if r['batch'] == n)
                        for k in ('wall_ms', 'gpu_interval_ms')} for n in (1, 2)}
    result = dict(scope='Real weights, fixed zero synthetic KV at length512; full dummy Talker+MTP path. '
                        'Excludes client, scheduler, sampling and natural prefix replay. GPU intervals can include launch gaps.',
                  warmups_per_batch=8, alternating_pairs=20, rows=records, medians=summary,
                  single_step_reduction_fraction=1-summary['1']['wall_ms']/summary['2']['wall_ms'],
                  aggregate_rate_ratio_single_over_double=summary['2']['wall_ms']/(2*summary['1']['wall_ms']),
                  all_outputs_finite=all(r['output_finite'] for r in records))
    with Path(target).open('x') as f:
        json.dump(result, f, indent=2)
    # The first actual execution clears model-side dummy-run state as usual.


@torch.inference_mode()
def calibrate_s16(runner, target):
    """Small shared cost table for the frozen workload's at most 8 audio streams."""
    for cache in runner.kv_caches:
        if isinstance(cache, torch.Tensor):
            cache.zero_()
    for name in ("input_ids", "inputs_embeds", "talker_mtp_input_ids",
                 "talker_mtp_inputs_embeds", "last_talker_hidden", "text_step"):
        getattr(runner, name).gpu[:8].zero_()
    rows=[]
    for context in (512,2048):
        runner.positions[:8].fill_(context-1)
        if runner.uses_mrope:
            runner.mrope_positions.gpu[:, :8].fill_(context-1)
        def forward(n):
            return runner._dummy_run(n, uniform_decode=True, force_attention=True,
                                    allow_microbatching=False, profile_seq_lens=context)
        for n in range(1,9):
            for _ in range(4):
                forward(n)
        for repeat in range(10):
            for n in (range(1,9) if repeat%2==0 else range(8,0,-1)):
                torch.cuda.synchronize()
                before=time.perf_counter()
                outputs=forward(n)
                torch.cuda.synchronize()
                wall_ms=(time.perf_counter()-before)*1000
                rows.append(dict(context=context,batch=n,repeat=repeat,wall_ms=wall_ms,
                                 finite=all(bool(torch.isfinite(t).all()) for t in outputs)))
    medians={str(context):{str(n):statistics.median(r['wall_ms'] for r in rows
                          if r['context']==context and r['batch']==n) for n in range(1,9)}
             for context in (512,2048)}
    result=dict(scope='Real Stage1 weights, synthetic fixed zero KV; startup empirical cost model, not measured user requests',
                rows=rows, medians_by_context_batch_ms=medians, all_outputs_finite=all(r['finite'] for r in rows))
    with Path(target).open('x') as f:json.dump(result,f,indent=2)
