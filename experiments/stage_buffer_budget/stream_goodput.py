"""Client-consumption streaming score; complete offered cohort, fixed arrival window.

Audio is evaluated in <=20 ms playable units to avoid chunk-size-dependent weights.
Text units are canonical visible tokenizer spans, never hidden model token IDs.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


def score_units(units, config):
    """units=(client arrival seconds, playable/read seconds, equivalent tokens)."""
    if not units:
        return {'utility':0.,'rebuffer_s':0.,'max_rebuffer_s':0.,'raw_units':0.,'net':0.}
    first=units[0][0]; end=first; content=0.; utility=0.; stalls=[]; peak=0.
    for arrival,duration,mass in units:
        if duration<=0 or mass<=0 or not all(math.isfinite(v) for v in (arrival,duration,mass)):
            raise ValueError('invalid client consumption unit')
        stall=max(0.,arrival-end)
        if stall:stalls.append(stall)
        lag=max(0.,arrival-(first+content))
        h=max(0.,1.-lag/config['lateness_decay_s'])
        end=max(arrival,end)+duration
        buffered=max(0.,end-arrival)
        w=max(0.,1.-max(0.,buffered-config['buffer_full_credit_s'])/config['buffer_decay_s'])
        utility+=mass*h*w
        content+=duration;peak=max(peak,buffered)
    rebuffer=sum(stalls)
    return {'utility':utility,'rebuffer_s':rebuffer,'max_rebuffer_s':max(stalls,default=0.),
        'raw_units':sum(u[2] for u in units),'net':max(0.,utility-config['mu']*rebuffer),
        'peak_client_buffer_s':peak,'content_s':content}


def request_score(row,result,config,tokenizer=None):
    path=row['request_path'];units=[]
    soft=config.get('metric_version')=='soft_ttft_v2'
    if result and (result.get('status')=='success' or soft):
        if path=='audio':
            for chunk in result.get('audio_chunks',[]):
                duration=chunk['duration_s']; n=math.ceil(duration/config['audio_unit_s'])
                for _ in range(n):units.append((chunk['arrival_monotonic_s'],duration/n,config['audio_equivalent_tps']*duration/n))
        else:
            if tokenizer is None:raise ValueError('visible text scoring requires the frozen tokenizer')
            text_path=Path(result['artifact_directory'])/'text.txt'
            text=text_path.read_text() if text_path.exists() else ''
            encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            for start,end in encoded['offset_mapping']:
                if end<=start:raise ValueError('token has no visible text span')
                idx=bisect.bisect_left(result['text_character_ends'],end)
                units.append((result['text_chunk_arrival_monotonic_s'][idx],1/config['text_tps'],1.))
    score=score_units(units,config)
    first=result.get('ttfp_s' if path=='audio' else 'ttft_s') if result else None
    limit=row['ttfp_slo_ms' if path=='audio' else 'ttft_slo_ms']/1000
    complete=bool(result and result.get('status')=='success' and result.get('done_received'))
    first_ok=first is not None and first<=limit
    score.update(request_id=row['source_request_id'],path=path,tier=row.get('slo_tier'),complete=complete,
      first_output_s=first,first_slo_s=limit,first_slo_met=first_ok,
      e2e_s=result.get('e2e_s') if result else None,
      e2e_slo_met=bool(complete and result['e2e_s']<=row['slo_ms']/1000),
      contribution=score['net'] if complete and first_ok else 0.)
    if soft:
        beta=float(config['beta'])
        if not math.isfinite(beta) or beta<0:raise ValueError('beta must be finite and nonnegative')
        violation=max(0.,first-limit) if first is not None else max(0.,float((result or {}).get('e2e_s') or 0.)-limit)
        score.update(first_violation_s=violation,beta=beta,first_penalty=beta*violation,
          contribution=max(0.,score['utility']-beta*violation-config['mu']*score['rebuffer_s']))
    return score


def main():
    p=argparse.ArgumentParser();p.add_argument('--workload',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-name',default='stream_score.json');p.add_argument('--tokenizer',default='/data/models/Qwen3-Omni-30B-A3B-Instruct')
    a=p.parse_args(); config=json.loads(a.config.read_text());manifest=json.loads((a.run/'manifest.json').read_text())
    if hashlib.sha256(a.workload.read_bytes()).hexdigest()!=manifest['workload_sha256']:raise ValueError('workload hash mismatch')
    if manifest.get('status')!='completed':raise ValueError('score requires a completely drained offered cohort')
    rows=[json.loads(l) for l in a.workload.read_text().splitlines() if l]
    results={r['source_request_id']:r for r in json.loads((a.run/'results.json').read_text())}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    scores=[request_score(r,results.get(r['source_request_id']),config,tokenizer) for r in rows]
    window=manifest['arrival_window_s']
    value={'scope':'pilot_not_formal','metric':('sum(positive(weighted_visible_units - beta*first_output_violation - mu*rebuffer)) / T' if config.get('metric_version')=='soft_ttft_v2' else 'sum(complete * first_output_SLO * positive(weighted_visible_units - mu*rebuffer)) / arrival_window'),
      'metric_config':config,'metric_config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),
      'workload_sha256':manifest['workload_sha256'],'arrival_window_s':window,'offered':len(rows),
      'G_stream':sum(s['contribution'] for s in scores)/window,'requests':scores}
    (a.run/a.output_name).write_text(json.dumps(value,indent=2))
    print(json.dumps({k:v for k,v in value.items() if k!='requests'},indent=2))

if __name__=='__main__':main()
