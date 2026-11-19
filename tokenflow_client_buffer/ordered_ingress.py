"""Source-clock dispatch, with an independent native enqueue order audit."""
import asyncio
import copy
import json
from pathlib import Path
import time


async def ordered_requests(rows, *, launch, trace_path, output, origin, timeout=30.):
    indexed = sorted(enumerate(rows), key=lambda pair: (pair[1].get('arrival_s', 0), pair[0]))
    ids = [row['payload']['omni_scheduling']['source_request_id'] for _, row in indexed]
    if len(ids) != len(set(ids)):
        raise ValueError('ordered benchmark requires unique source_request_id values')
    launches = {}
    tasks = []
    async def dispatch(row, rid, rank):
        await asyncio.sleep(max(0., origin + row.get('arrival_s', 0) - time.monotonic()))
        payload_row = copy.deepcopy(row)
        payload_row['payload']['omni_scheduling']['ingress_order'] = rank
        launches[rid] = time.monotonic()
        return await launch(payload_row)

    with Path(trace_path).open() as trace, (Path(output)/'ingress_order.jsonl').open('w', buffering=1) as log:
        trace.seek(0, 2)
        try:
            tasks = [asyncio.create_task(dispatch(row, rid, rank))
                     for rank, ((_, row), rid) in enumerate(zip(indexed, ids))]
            known = set(ids)
            scanned = 0
            for rank, ((row_index, row), rid) in enumerate(zip(indexed, ids)):
                scheduled = origin + row.get('arrival_s', 0)
                deadline = max(scheduled, time.monotonic()) + timeout
                while True:
                    position = trace.tell()
                    line = trace.readline()
                    if line and not line.endswith('\n'):
                        trace.seek(position)
                        line = ''
                    if line.strip():
                        scanned += 1
                        if scanned % 128 == 0:
                            await asyncio.sleep(0)
                        event = json.loads(line)
                        if event.get('event') != 'stage_enqueue' or str(event.get('stage_id')) != '0':
                            continue
                        source = event.get('sched_source_request_id')
                        if source not in known:
                            continue
                        if source != rid:
                            raise RuntimeError(f'engine ingress order mismatch: expected {rid}, got {source}')
                        queued = event['monotonic']
                        launched = launches[rid]
                        if queued < launched:
                            raise RuntimeError('stale engine enqueue acknowledgement')
                        log.write(json.dumps(dict(rank=rank, workload_row=row_index,
                            request_id=rid, engine_request_id=event.get('request_id'),
                            scheduled_monotonic=scheduled, launched_monotonic=launched,
                            enqueued_monotonic=queued, acknowledged_monotonic=time.monotonic(),
                            origin=origin, admission_event='stage_enqueue', stage_id=0))+'\n')
                        break
                    if tasks[rank].done():
                        if tasks[rank].cancelled():
                            raise RuntimeError(f'request cancelled before ingress acknowledgement: {rid}')
                        if tasks[rank].exception() is not None:
                            raise tasks[rank].exception()
                        raise RuntimeError(f'request ended without stage-0 ingress acknowledgement: {rid}')
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'no stage-0 ingress acknowledgement for {rid}')
                    await asyncio.sleep(.005)
            results = await asyncio.gather(*tasks)
            by_row = {index: result for (index, _), result in zip(indexed, results)}
            return [by_row[i] for i in range(len(rows))]
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
