"""Local experiment ingress and authoritative W ledger.

Output is forwarded immediately. Only client acknowledgements fund pauses.
Server-stream parsing supplies a separate production trace for calibration.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

from aiohttp import ClientSession, ClientTimeout, web

from .core import CapacityEstimate, CapacityEstimator, ClientFeedback, WorkingSet, stream_contract
from .stream import StreamDecoder, atomic_json, clock_domain, special_ids, sse_events


class Gateway:
    def __init__(self, config):
        self.config = config
        self.modality = config["modality"]
        self.session = config["session"]
        self.clock_domain = clock_domain()
        self.w = WorkingSet(self.modality, max_inflight=config.get("max_inflight", 32),
                            recovery_samples=config.get("recovery_samples", 1))
        self.feedback = {}
        self.ready = {}
        self.special = special_ids(config["model"])
        self.snapshot = Path(config["snapshot_path"])
        self.log = Path(config["gateway_log"]).open("a", buffering=1)
        self.est = None
        self.profile_mtime = None
        self.last_publish = 0.
        self.last_status = None
        self.stopping = False
        self.estimator = CapacityEstimator(self.modality, safety=config.get("capacity_safety", .85),
            ttl_s=config.get("capacity_ttl_s", 30.), min_window_s=2.)
        self.window_start = time.monotonic()
        self.window_units = 0.
        self.previous_telemetry = None

    def event(self, name, **values):
        self.log.write(json.dumps(dict(event=name, monotonic=time.monotonic(), **values), allow_nan=False) + "\n")

    def capacity(self, now):
        path = self.config.get("capacity_path")
        if path:
            try:
                p = Path(path)
                stamp = p.stat().st_mtime_ns
                if stamp != self.profile_mtime:
                    raw = json.loads(p.read_text())
                    if raw["modality"] != self.modality:
                        raise ValueError("capacity unit mismatch")
                    self.est = CapacityEstimate(**raw)
                    self.profile_mtime = stamp
                    self.estimator.estimate = self.est
                    self.event("capacity_updated", estimate=raw)
            except (OSError, ValueError, KeyError, TypeError):
                self.est = None
        return self.est

    def publish(self):
        now = time.monotonic()
        est = self.capacity(now)
        state = self.w.snapshot(est, now, native=self.config["arm"] == "fcfs")
        if self.config.get("capacity_only"):
            # Ablation: retain the same W/Gamma gate but run native FCFS.
            state.update(mode="fcfs", reason="capacity_only")
        state.update(schema_version=1, session=self.session, modality=self.modality,
                     clock_domain=self.clock_domain, written_at=now,
                     capacity_valid_until=est.valid_until if est else 0.,
                     feedback={rid: asdict(f) for rid, f in self.feedback.items()
                               if rid in state["members"]})
        atomic_json(self.snapshot, state)
        status = (state["mode"], state["reason"], state["demand"], state["gamma"], state["waiting"])
        if status != self.last_status or now - self.last_publish > 1:
            self.event("working_set", **{k:v for k,v in state.items() if k != "feedback"})
            self.last_status, self.last_publish = status, now

    def update_capacity(self, now):
        if now-self.window_start < 2.:
            return
        try:
            telemetry = json.loads(Path(self.config["execution_telemetry_path"]).read_text())
            previous = self.previous_telemetry
            self.previous_telemetry = telemetry
            if previous and self.est:
                steps = telemetry["steps"]-previous["steps"]
                full = telemetry["full_steps"]-previous["full_steps"]
                holds = telemetry["held_steps"]-previous["held_steps"]
                saturated = (steps > 0 and full/steps >= .9 and holds == 0
                    and 0 <= now-telemetry["written_at"] <= .5
                    and telemetry["session"] == self.session)
                before = self.est
                self.est = self.estimator.observe(generated_units=self.window_units,
                    duration_s=now-self.window_start, now=now, saturated=saturated,
                    profile_id=self.est.profile_id)
                self.event("capacity_window", units=self.window_units, duration_s=now-self.window_start,
                    full_steps=full, steps=steps, holds=holds, saturated=saturated,
                    estimate=asdict(self.est))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.window_start, self.window_units = now, 0.

    async def loop(self):
        while not self.stopping:
            now = time.monotonic()
            self.capacity(now)
            self.update_capacity(now)
            while True:
                rid = self.w.admit_next(self.capacity(now), now, native=self.config["arm"] == "fcfs")
                if rid is None:
                    break
                self.event("admitted", request_id=rid, rate=self.w.requests[rid].rate)
                self.ready[rid].set()
            self.publish()
            await asyncio.sleep(.025)

    async def contract(self, request):
        return web.json_response(dict(session=self.session, modality=self.modality,
                                      clock_domain=self.clock_domain, arm=self.config["arm"]))

    async def receive_feedback(self, request):
        try:
            raw = await request.json()
            rid = raw["request_id"]
            if rid not in self.w.requests:
                return web.json_response({"error":"request not registered"}, status=409)
            f = ClientFeedback.parse(raw, now=time.monotonic(), session=self.session,
                    modality=self.modality, clock_domain=self.clock_domain,
                    rate=self.w.requests[rid].rate, previous=self.feedback.get(rid))
            self.feedback[rid] = f
            self.event("client_feedback", **asdict(f))
            return web.json_response({"accepted":True})
        except (ValueError, KeyError, TypeError) as e:
            self.event("feedback_rejected", error=str(e))
            return web.json_response({"error":str(e)}, status=400)

    async def stream(self, request):
        try:
            payload = await request.json()
            rid, rate = stream_contract(payload, self.modality)
            if request.headers.get("X-TokenFlow-Session") != self.session:
                raise ValueError("client session mismatch")
            self.w.add(rid, rate)
            self.ready[rid] = asyncio.Event()
            # Arrival order is server-owned, never supplied for priority gaming.
            payload["omni_scheduling"]["ingress_order"] = self.w.requests[rid].arrival_order
            self.event("arrived", request_id=rid, rate=rate, payload=payload)
        except (ValueError, KeyError, TypeError) as e:
            return web.json_response({"error":str(e)}, status=400)
        response = None
        status = "cancelled"
        try:
            while not self.ready[rid].is_set():
                if request.transport is None or request.transport.is_closing():
                    raise ConnectionResetError("client disconnected while queued")
                try:
                    await asyncio.wait_for(self.ready[rid].wait(), .1)
                except asyncio.TimeoutError:
                    pass
            decoder = StreamDecoder(self.modality, self.special)
            async with self.http.post(self.config["upstream"] + "/v1/chat/completions", json=payload) as upstream:
                if upstream.status != 200:
                    error = await upstream.text()
                    self.event("upstream_error", request_id=rid, status=upstream.status, error=error)
                    status = "error"
                    return web.Response(status=upstream.status, text=error)
                response = web.StreamResponse(status=200, headers={"Content-Type":"text/event-stream"})
                await response.prepare(request)
                async for data in sse_events(upstream.content):
                    units, _ = decoder.event(data)
                    if units:
                        self.window_units += units
                        self.event("server_output", request_id=rid, units=units,
                                   cumulative_units=decoder.units, modality=self.modality)
                    await response.write(("data: " + data + "\n\n").encode())
                if not decoder.done or decoder.units <= 0:
                    raise ValueError("incomplete or empty upstream stream")
                await response.write_eof()
                status = "success"
                self.event("generation_complete", request_id=rid, units=decoder.units,
                           content_sha256=decoder.hash.hexdigest())
                return response
        except (ConnectionError, asyncio.CancelledError):
            raise
        except Exception as e:
            status = "error"
            self.event("request_error", request_id=rid, error=repr(e))
            if response is None:
                return web.json_response({"error":repr(e)}, status=500)
            try:
                await response.write(("data: " + json.dumps({"error":repr(e)}) + "\n\n").encode())
                await response.write_eof()
            except ConnectionError:
                pass
            return response
        finally:
            self.w.complete(rid)
            self.event("terminal", request_id=rid, status=status)
            self.publish()

    async def startup(self, app):
        self.http = ClientSession(timeout=ClientTimeout(total=600), connector=__import__("aiohttp").TCPConnector(limit=0))
        self.task = asyncio.create_task(self.loop())

    async def cleanup(self, app):
        self.stopping = True
        await self.task
        await self.http.close()
        self.log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--port", type=int, default=8096)
    args = p.parse_args()
    g = Gateway(json.loads(Path(args.config).read_text()))
    app = web.Application(client_max_size=16*1024*1024)
    app.router.add_get("/contract", g.contract)
    app.router.add_post("/feedback", g.receive_feedback)
    app.router.add_post("/v1/chat/completions", g.stream)
    app.on_startup.append(g.startup)
    app.on_cleanup.append(g.cleanup)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__": main()
