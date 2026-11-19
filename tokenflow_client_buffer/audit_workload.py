"""Audit completed paired audio workloads from real stage and client traces."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .stream import audio_info
from .workload_contract import expected_chunks


def read_events(root, arm):
    w = json.loads((root/'runtime'/f'workload_{arm}_window.json').read_text())
    if 'end_offset' not in w:
        raise ValueError(f'{arm}: complete generation/drain window missing')
    with (Path(w['trace'])/'stage_events.jsonl').open('rb') as f:
        f.seek(w['start_offset'])
        return [json.loads(line) for line in f.read(w['end_offset']-w['start_offset']).splitlines()]


def wire_counts(path):
    chunks = []
    done = False
    with path.open() as wire:
        for line in wire:
            if not line.startswith('data: '):
                continue
            data = line[6:].strip()
            if data == '[DONE]':
                done = True
                continue
            obj = json.loads(data)
            for choice in obj.get('choices', []):
                delta = choice.get('delta') or {}
                content = delta.get('content')
                raw = content if obj.get('modality') == 'audio' else (delta.get('audio') or {}).get('data')
                if raw:
                    duration, _, rate, channels = audio_info(base64.b64decode(raw, validate=True))
                    chunks.append(dict(samples=round(duration*rate), rate=rate, channels=channels))
    if not done or not chunks:
        raise ValueError('Incomplete audio wire stream')
    return chunks


def summarize_events(events, source_ids):
    mapping = {}
    for e in events:
        if e['event'] == 'client_buffer_contract' and e.get('source_request_id') in source_ids:
            for k in ('request_id', 'logical_request_id', 'execution_id'):
                if e.get(k):
                    mapping[e[k]] = e['source_request_id']
    grouped = defaultdict(list)
    for e in events:
        rid = mapping.get(e.get('logical_request_id')) or mapping.get(e.get('request_id'))
        if rid:
            grouped[rid].append(e)
    return grouped


def audit(root):
    root = Path(root)
    targets = {r['request_id']: r for r in json.loads((root/'workloads/targets.json').read_text())}
    out = dict(passed=False, scope='Full short streams; exact workload sizes, not performance', policies={}, errors=[])
    expected_workload = hashlib.sha256((root/'workloads/audio_0.jsonl').read_bytes()).hexdigest()
    manifests = []
    for arm in ('fcfs', 'buffer'):
        case = root/'runs'/f'workload_{arm}'
        manifests.append(json.loads((case/'service_manifest.json').read_text()))
        if not json.loads((case/'drain_after.json').read_text()).get('drained'):
            out['errors'].append(f'{arm}: service not drained')
        client_manifest = json.loads((case/'client/manifest.json').read_text())
        if client_manifest['workload_sha256'] != expected_workload:
            out['errors'].append(f'{arm}: workload hash mismatch')
        rows = {r['request_id']: r for r in json.loads((case/'client/results.json').read_text())}
        if set(rows) != set(targets):
            out['errors'].append(f'{arm}: request set mismatch')
        events = read_events(root, arm)
        grouped = summarize_events(events, targets)
        result = {}
        for rid, target in targets.items():
            errors = []
            es = grouped.get(rid, [])
            thinker = [e for e in es if e['event'] == 'stage_study_thinker_output_produced']
            frames = [e['codec_total_frames'] for e in es if e['event'] == 'stage_study_codec_frame_produced']
            begins = [e for e in es if e['event'] == 'stage_study_chunk_conversion_started']
            ends = [e for e in es if e['event'] == 'stage_study_chunk_conversion_completed']
            text = [e for e in es if e['event'] == 'stage_study_text_conditioning_selected']
            if not thinker or not frames or not ends or not text:
                errors.append('missing stage evidence')
            output = max((e['generated_output_tokens_total'] for e in thinker), default=0)
            if output != target['thinker_output_tokens']:
                errors.append('Thinker output target mismatch')
            if frames != list(range(1, target['expected_codec_frames']+1)):
                errors.append('codec frame target/sequence mismatch')
            fields = ('chunk_id', 'new_frames', 'left_context_frames', 'codec_payload_frames', 'codec_total_frames')
            signature = lambda e: tuple(e[k] for k in fields)
            if any(e['scheduled_codec_elements'] != e['codec_payload_frames']*16 for e in begins):
                errors.append('codec element count differs from 16 codebooks per frame')
            if [signature(e) for e in begins] != [signature(e) for e in ends]:
                errors.append('decoder start/completion mismatch')
            geometry = expected_chunks(target['expected_codec_frames'], target['codec_chunk_frames'], target['codec_left_context_frames'])
            if [{k: e[k] for k in geometry[0]} for e in ends] != geometry:
                errors.append('decoder chunk/context geometry mismatch')
            if [e['chunk_id'] for e in ends] != list(range(len(ends))):
                errors.append('duplicate/missing decoder chunk')
            row = rows.get(rid, {})
            if row.get('status') != 'success':
                errors.append('client did not complete successfully')
            key = hashlib.sha256(rid.encode()).hexdigest()[:16]
            wire = wire_counts(case/'client'/key/'stream.sse')
            if [c['samples'] for c in wire] != [e['audio_samples'] for e in ends]:
                errors.append('decoded/client waveform sample mismatch')
            if row.get('startup_s') != row.get('first_ready_s'):
                errors.append('unexpected playback startup gate')
            positions = [i for e in text for i in range(e['start_index'], e['end_index'])]
            result[rid] = dict(thinker_prompt_tokens=sorted({e['prompt_tokens'] for e in thinker}),
                thinker_output_tokens=output, codec_frames=len(frames), decoder_chunks=len(ends),
                decoder_payload_frames=sum(e['codec_payload_frames'] for e in ends),
                decoder_new_frames=sum(e['new_frames'] for e in ends),
                decoder_context_frames=sum(e['left_context_frames'] for e in ends),
                decoder_audio_samples=sum(e['audio_samples'] for e in ends),
                client_audio_samples=sum(c['samples'] for c in wire),
                audio_sample_rates=sorted({c['rate'] for c in wire}),
                audio_channels=sorted({c['channels'] for c in wire}),
                talker_selected_text_positions=positions,
                talker_conditioning_counts=dict(Counter(e['conditioning_kind'] for e in text)),
                talker_prefill_input_shapes=[{k:e.get(k) for k in ('assistant_hidden_rows','first_text_prefill_index','first_text_source')} for e in text if e['phase']=='prefill'],
                waveform_sha256=row.get('content_sha256'), duration_s=row.get('received_units'), errors=errors)
            out['errors'].extend(f'{arm}/{rid}: {s}' for s in errors)
        out['policies'][arm] = result
    if manifests[0] != manifests[1]:
        out['errors'].append('different model service/configuration between arms')
    out['workload_sha256'] = expected_workload
    out['same_loaded_service'] = manifests[0] == manifests[1]
    mismatches = []
    for rid in targets:
        a, b = (out['policies'][arm][rid] for arm in ('fcfs','buffer'))
        for key in a:
            if key not in ('waveform_sha256','errors','duration_s') and a[key] != b[key]:
                mismatches.append(dict(request_id=rid, metric=key, fcfs=a[key], buffer=b[key]))
    out['paired_mismatches'] = mismatches
    out['paired_waveform_hashes_equal'] = all(out['policies']['fcfs'][rid]['waveform_sha256'] == out['policies']['buffer'][rid]['waveform_sha256'] for rid in targets)
    out['passed'] = not out['errors'] and not mismatches
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    path = args.root/'reports/workload_audit.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'policies'}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
