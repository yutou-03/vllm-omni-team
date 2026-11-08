"""Open-loop benchmark with actual receive queues and paced simulated sinks."""
import argparse
import asyncio
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from aiohttp import ClientSession, ClientTimeout, TCPConnector

from .core import ConsumerBuffer, rate_field, stream_contract
from .stream import StreamDecoder, clock_domain, special_ids, sse_events


async def consume_request(row, *, session, endpoint, contract, output, special, origin,
                          feedback_interval=.1, timeout=600.):
    rid, rate = stream_contract(row["payload"], contract["modality"])
    scheduled = origin + row.get("arrival_s", 0)
    await asyncio.sleep(max(0, scheduled - time.monotonic()))
    directory = output / hashlib.sha256(rid.encode()).hexdigest()[:16]
    directory.mkdir()
    log = (directory / "events.jsonl").open("w", buffering=1)
    def event(name, **values):
        log.write(json.dumps(dict(event=name, monotonic=time.monotonic(), **values), allow_nan=False) + "\n")
    buffer = ConsumerBuffer(rate, row.get("startup_s", 0.) * rate)
    queue = deque()
    decoder = StreamDecoder(contract["modality"], special)
    start = time.monotonic()
    stopped = False
    received_done = asyncio.Event()
    feedback_errors = []
    last_feedback = 0.
    consumed_cursor = 0.
    result = dict(request_id=rid, modality=contract["modality"], rate=rate,
                  scheduled_at=scheduled, sent_at=start, send_lag_s=start-scheduled,
                  status="error", workload=row)

    async def sink():
        nonlocal last_feedback, consumed_cursor
        while not stopped:
            now = time.monotonic()
            buffer.tick(now)
            consume = buffer.consumed - consumed_cursor
            consumed_cursor = buffer.consumed
            while consume > 1e-9 and queue:
                units, payload = queue[0]
                used = min(consume, units)
                consume -= used
                if units - used <= 1e-9:
                    queue.popleft()
                else:
                    queue[0] = (units-used, payload)
            if now - last_feedback >= feedback_interval or buffer.completed_at is not None:
                raw = buffer.feedback(now, request_id=rid, session=contract["session"],
                        modality=contract["modality"], clock_domain=clock_domain())
                try:
                    async with session.post(endpoint + "/feedback", json=raw, timeout=ClientTimeout(total=2)) as reply:
                        text = await reply.text()
                        if reply.status not in (200, 409):
                            feedback_errors.append(text)
                except Exception as error:
                    feedback_errors.append(repr(error))
                last_feedback = now
                event("client_buffer", **raw, queued_units=sum(x[0] for x in queue))
            if buffer.completed_at is not None:
                return
            await asyncio.sleep(.02)

    task = asyncio.create_task(sink())
    try:
        async with asyncio.timeout(timeout):
            with (directory / "stream.sse").open("w") as wire:
                async with session.post(endpoint + "/v1/chat/completions", json=row["payload"],
                        headers={"X-TokenFlow-Session":contract["session"]}) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
                    async for data in sse_events(response.content):
                        arrived = time.monotonic()
                        wire.write("data: " + data + "\n\n")
                        # Audio is credited only after decode completes.
                        units, retained = decoder.event(data)
                        ready = time.monotonic()
                        if units:
                            buffer.tick(ready)
                            before = buffer.buffered
                            queue.append((units, retained))
                            buffer.receive(units, ready)
                            event("received", units=units, buffered_before=before,
                                  arrived_at=arrived, ready_at=ready,
                                  total_received=buffer.received)
                        await asyncio.sleep(0)
            result["generation_done_at"] = time.monotonic()
            if not decoder.done or buffer.received <= 0:
                raise ValueError("missing DONE or no consumed-modality content")
            buffer.finish(time.monotonic())
            received_done.set()
            await task
            if feedback_errors:
                raise RuntimeError("feedback errors: " + str(feedback_errors[:3]))
            if queue or abs(buffer.received-buffer.consumed) > 1e-7:
                raise ValueError("client buffer failed conservation/drain")
            result["status"] = "success"
    except Exception as error:
        result["error"] = repr(error)
    finally:
        stopped = True
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        result.update(buffer=asdict(buffer), received_units=buffer.received,
                      content_sha256=decoder.hash.hexdigest(), text=decoder.text,
                      text_token_ids=decoder.text_token_ids, audio_chunks=decoder.audio_chunks,
                      startup_s=buffer.started_at-scheduled if buffer.started_at is not None else None,
                      first_ready_s=buffer.first_ready-scheduled if buffer.first_ready is not None else None,
                      stall_s=buffer.stalled_s, max_stall_s=buffer.max_stall_s,
                      completed_at=buffer.completed_at, ended_at=time.monotonic(),
                      feedback_errors=feedback_errors, remaining_queue=len(queue))
        (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        log.close()
    print(json.dumps({k: result.get(k) for k in ("request_id","status","startup_s","stall_s","received_units","error")}), flush=True)
    return result


async def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(s) for s in Path(args.workload).read_text().splitlines() if s]
    async with ClientSession(timeout=ClientTimeout(total=args.timeout), connector=TCPConnector(limit=0)) as session:
        async with session.get(args.endpoint + "/contract") as r:
            contract = await r.json()
        if contract["clock_domain"] != clock_domain():
            raise ValueError("pilot requires client/server shared host clock")
        origin = time.monotonic()+.5
        (output / "manifest.json").write_text(json.dumps(dict(contract=contract, origin=origin,
            workload_sha256=hashlib.sha256(Path(args.workload).read_bytes()).hexdigest(),
            sink="paced simulator with retained payload queue; no physical speaker", args=vars(args)), indent=2))
        special = special_ids(args.model)
        results = await asyncio.gather(*(consume_request(row, session=session, endpoint=args.endpoint,
            contract=contract, output=output, special=special, origin=origin, timeout=args.timeout)
            for row in rows))
        (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))
        return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:8096")
    p.add_argument("--model", default="/data/models/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--timeout", type=float, default=600)
    args = p.parse_args()
    results = asyncio.run(run(args))
    if any(r["status"] != "success" for r in results):
        raise SystemExit(1)


if __name__ == "__main__": main()
