"""Audit text delivery conservation and multiple real KV-preserving hold chains."""
import argparse,json,hashlib
from pathlib import Path

def audit(path):
    holds={};chains=[];active=set();peak=0;errors=[];final={};generated={};min_stock={}
    for line in path.open():
        e=json.loads(line);n=e['event'];t=e['monotonic'];rid=e.get('request_id')
        if n=='small_hold_requested':
            c={'victim':rid,'requested':t,'urgent':e['evidence'].get('urgent_request_id'),'evidence':e['evidence'],'kv_before':e['kv_block_ids']}
            holds[rid]=c;chains.append(c);active.add(rid);peak=max(peak,len(active))
        elif n=='urgent_keep_admitted' and e['victim_request_id'] in holds:
            holds[e['victim_request_id']].update(admitted=t,kv_at_admission=e['victim_kv_block_ids'])
        elif n=='small_hold_released' and rid in holds:holds[rid]['released']=t
        elif n=='small_hold_first_rescheduled' and rid in holds:
            holds[rid]['rescheduled']=t;active.discard(rid)
        elif n=='stage_buffer_text_inventory':
            if e['received_chars']-e['released_chars']!=e['buffered_chars'] or e['buffered_chars']<0:errors.append('character conservation '+rid)
            if e['closed']:
                final[rid]=e
                if e['action']!='completed' or e['received_wire_sha256']!=e['released_wire_sha256'] or e['received_text_sha256']!=e['released_text_sha256']:
                    errors.append('delivery incomplete or reordered '+rid)
            for c in chains:
                logical=c['evidence'].get('delivery_logical_request_id',c['victim'])
                if logical!=rid:continue
                if c['requested']<=t and 'first_new_chunk' not in c:
                    key=c['victim'];min_stock[key]=min(min_stock.get(key,float('inf')),e['buffered_tokens'])
                old=generated.get(rid,e['received_chars'])
                if e['action']=='received' and 'rescheduled' in c and t>=c['rescheduled'] and e['received_chars']>old:
                    c.setdefault('first_new_chunk',t)
            generated[rid]=e['received_chars']
    for c in chains:
        keys=['requested','admitted','released','rescheduled','first_new_chunk']
        c['complete']=all(k in c for k in keys) and all(c[a]<=c[b] for a,b in zip(keys,keys[1:]))
        c['kv_retained']=c.get('kv_at_admission')==c['kv_before']
        if 'rescheduled' in c:c['actual_pause_ms']=1000*(c['rescheduled']-c['requested'])
        if 'first_new_chunk' in c:c['resume_to_first_chunk_ms']=1000*(c['first_new_chunk']-c['rescheduled'])
    return {'scope':'mechanism_only','requested':len(chains),'complete_chains':sum(c['complete'] for c in chains),
        'kv_retained_chains':sum(c['kv_retained'] for c in chains),'peak_concurrent_holds':peak,
        'minimum_owned_text_tokens_by_victim':min_stock,'delivery_streams':len(final),'integrity_errors':errors,'chains':chains,
        'delivery':{k:{x:v[x] for x in ['received_chars','released_chars','received_tokens','released_tokens','received_text_sha256','released_text_sha256']} for k,v in final.items()}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=audit(a.trace);a.output.write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k not in ['chains','delivery']}))
