"""Audit receiver-owned inventory and actual pause/admission/resume chains."""
import argparse,collections,json
from pathlib import Path


def audit(events):
    events = sorted(events, key=lambda e: e['monotonic'])
    enqueued=collections.defaultdict(list);released=collections.defaultdict(list);maxima={1:0,2:0}
    chains=[];current={};sent=collections.defaultdict(set);receive=collections.defaultdict(set);consume=collections.defaultdict(set)
    def key(e):return (e.get('physical_request_id',e.get('request_id')),e.get('pipeline_attempt_id',0),e.get('stage_attempt_id',0))
    for e in events:
        n=e['event']; k=key(e);sid=e.get('stage_id');t=e['monotonic']
        if n=='stage_study_receiver_buffer_enqueued':
            enqueued[(sid,k)].append(e['chunk_id']);maxima[sid]=max(maxima[sid],e['buffered_units'])
        elif n=='stage_study_receiver_buffer_released':released[(sid,k)].append(e['chunk_id'])
        if n in ('stage_study_text_payload_sent','stage_study_chunk_sent'):sent[(sid,k)].add(e['chunk_id'])
        if n in ('stage_study_text_payload_received','stage_study_chunk_received'):receive[(sid-1,k)].add(e['chunk_id'])
        if n=='stage_study_chunk_conversion_started':consume[(1,k)].add(e['chunk_id'])
        if n=='small_hold_requested':
            c={'stage_id':sid,'victim':e['request_id'],'requested':t,'request_event':e};chains.append(c);current[(sid,e['request_id'])]=c
        elif n=='stage_first_forward':
            for (stage,victim),c in current.items():
                expected=c['request_event'].get('evidence',{}).get('urgent_request_id')
                if sid==stage and e.get('request_id')==expected and t>=c['requested']:
                    c.setdefault('admitted',t)
                    c.setdefault('admission_evidence','stage_first_forward')
                    c['urgent']=expected
                    c['urgent_first_batch_num_seqs']=e.get('batch_num_seqs')
        elif n=='urgent_keep_admitted':
            c=current.get((sid,e['victim_request_id']))
            if c:c.update(urgent=e['urgent_request_id'],admitted=t,victim_kv=e.get('victim_kv_block_ids'),admission_evidence='urgent_keep_admitted')
        elif n in ('small_hold_released','small_hold_first_rescheduled'):
            c=current.get((sid,e['request_id']))
            if c:c['released' if n=='small_hold_released' else 'rescheduled']=t
        elif n in ('stage_study_text_payload_sent','stage_study_chunk_sent'):
            c=current.get((sid,e.get('physical_request_id',e.get('request_id'))))
            if c and 'rescheduled' in c and t>=c['rescheduled'] and 'first_new_chunk' not in c:c['first_new_chunk']=t
    errors=[]
    for k,ids in enqueued.items():
        if ids!=list(range(len(ids))):errors.append(f'enqueue discontinuity {k}')
        out=released[k]
        if out!=ids[:len(out)]:errors.append(f'release discontinuity {k}')
    for k,ids in receive.items():
        if ids-sent[k]:errors.append(f'received without sent {k}: {sorted(ids-sent[k])}')
    for k,ids in consume.items():
        if ids-receive[k]:errors.append(f'consumed without received {k}')
    for c in chains:
        c['complete_chain']=all(k in c for k in ('admitted','released','rescheduled','first_new_chunk'))
        if c['complete_chain']:
            c['complete_chain'] = c['requested'] <= c['admitted'] <= c['released'] <= c['rescheduled'] <= c['first_new_chunk']
            c['actual_to_rescheduled_ms']=1000*(c['rescheduled']-c['requested'])
            c['release_to_first_chunk_ms']=1000*(c['first_new_chunk']-c['released'])
            c['reschedule_to_first_chunk_ms']=1000*(c['first_new_chunk']-c['rescheduled'])
    return {'scope':'mechanism_only','max_buffered_units_by_receiver_stage':maxima,'integrity_errors':errors,
      'complete_chains':sum(c['complete_chain'] for c in chains),'chains':chains,
      'buffered_streams':len(enqueued),'enqueued_chunks':sum(map(len,enqueued.values())),
      'released_chunks':sum(map(len,released.values()))}


def main():
 p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.trace.open() as f:events=[json.loads(l) for l in f if l.endswith('\n')]
 result=audit(events);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
