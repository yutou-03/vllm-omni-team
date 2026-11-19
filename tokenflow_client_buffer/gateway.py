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

from .core import buffer_conservativeness, CapacityEstimate, CapacityEstimator, ClientFeedback, WorkingSet, stream_contract
from .leases import PauseAdmission
from .admission import AdmissionSafety
from .stream import StreamDecoder, atomic_json, clock_domain, special_ids, sse_events


from .ordered_backend import OrderedBackend


class Gateway(PauseAdmission, AdmissionSafety):
    def __init__(self, config):
        self.config = config
        self.buffer_conservativeness = buffer_conservativeness(config.get("buffer_conservativeness", 1.0))
        self.ordered_backend = OrderedBackend(config["ordered_backend_trace"], event=self.event) if config.get("ordered_backend_trace") else None
        self.init_pause_admission()
        self.admission_safety_init()
        self.modality = config["modality"]
        self.session = config["session"]
        self.clock_domain = clock_domain()
        self.w = WorkingSet(self.modality, max_inflight=config.get("max_inflight", 32),
                            recovery_samples=config.get("recovery_samples", 1),
                            native_execution_limit=config.get("native_execution_limit", 1))
        self.w.admission_guard = self.admission_safety
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
        self.estimator = CapacityEstimator(self.modality, safety=config.get("capacity_safety", 1.0),
            ttl_s=config.get("capacity_ttl_s", 30.),
            min_window_s=config.get("capacity_window_s", 8. if self.modality == "audio" else 2.))
        self.window_start = time.monotonic()
        self.window_units = 0.
        self.previous_telemetry = None
        self.last_capacity_check = self.window_start

    def event(self, name, **values):
        self.log.write(json.dumps(dict(event=name, monotonic=time.monotonic(), **values), allow_nan=False) + "\n")

    def engine_queue_enabled(self):
        coordinator = getattr(self, 'coordinator', None)
        configured = self.config.get('engine_queue_admission',
            coordinator.config.get('engine_queue_admission', False) if coordinator else False)
        return self.config['arm'] == 'buffer' and configured is True

    def engine_queue_snapshot(self):
        enabled = self.engine_queue_enabled()
        return dict(engine_queue_admission=enabled,
                    admission_scope='target_output_stage' if enabled else 'http_ingress',
                    registered_ids=[rid for rid, r in self.w.requests.items()
                                    if r.state != 'finished'] if enabled else [])

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
        state.update(buffer_conservativeness=self.buffer_conservativeness, schema_version=1, session=self.session, modality=self.modality,
                     clock_domain=self.clock_domain, written_at=now,
                     capacity_valid_until=est.valid_until if est else 0.,
                     feedback={rid: asdict(f) for rid, f in self.feedback.items()
                               if rid in state["members"] or rid in state["paused_members"]},
                     **self.pause_snapshot(now), **self.engine_queue_snapshot())
        atomic_json(self.snapshot, state)
        status = (state["mode"], state["reason"], state["demand"], state["gamma"], state["waiting"])
        if status != self.last_status or now - self.last_publish > 1:
            self.event("working_set", **{k:v for k,v in state.items() if k != "feedback"})
            self.last_status, self.last_publish = status, now

    def update_capacity(self, now):
        from .capacity_signals import execution_window_evidence
        if (now-self.window_start < self.estimator.min_window_s
                or now-self.last_capacity_check < 2.):
            return
        self.last_capacity_check = now
        defer_empty = False
        telemetry = None
        try:
            telemetry = json.loads(Path(self.config["execution_telemetry_path"]).read_text())
            previous = self.previous_telemetry
            if previous and self.est:
                steps = telemetry["steps"]-previous["steps"]
                full = telemetry["full_steps"]-previous["full_steps"]
                holds = telemetry["held_steps"]-previous["held_steps"]
                duration = now-self.window_start
                measurement_valid = (steps > 0
                    and 0 <= now-telemetry["written_at"] <= .5
                    and telemetry["session"] == self.session)
                pending = next((r for r in self.w.requests.values() if r.state == 'waiting'), None)
                capacity_limited = (pending is not None and bool(self.w.members)
                    and self.w.demand + pending.rate > self.est.admission_bound)
                evidence = execution_window_evidence(telemetry, previous, duration)
                saturated = measurement_valid and evidence['saturated']
                buffer_steps = telemetry.get("buffer_steps", 0) - previous.get("buffer_steps", 0)
                fcfs_steps = telemetry.get("fcfs_steps", 0) - previous.get("fcfs_steps", 0)
                policy = ("buffer" if buffer_steps == steps else
                          "fcfs" if fcfs_steps == steps else "mixed")
                before = self.est
                # Silence belongs to the output window whether 1 or 32
                # requests execute. Never shorten a chunk's denominator.
                defer_empty = self.window_units == 0
                if measurement_valid:
                    self.est = self.estimator.observe(generated_units=self.window_units,
                        duration_s=duration, now=now, saturated=saturated,
                        profile_id=self.est.profile_id, policy=policy)
                if not measurement_valid:
                    evidence['saturation_basis'] = 'no_valid_execution'
                self.event("capacity_window", units=self.window_units, duration_s=duration,
                    full_steps=full, steps=steps, holds=holds, saturated=saturated,
                    policy=policy, buffer_steps=buffer_steps, fcfs_steps=fcfs_steps,
                    observed_output_per_s=self.window_units/duration,
                    capacity_limited=capacity_limited,
                    **{k:v for k,v in evidence.items() if k!='saturated'},
                    updated=self.est is not before, deferred_empty_window=defer_empty,
                    capacity_scope='current_delivered_service; no peak extrapolation; policy costs remain in wall time',
                    estimate=asdict(self.est))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if not defer_empty:
            self.previous_telemetry = telemetry
            self.window_start, self.window_units = now, 0.

    async def loop(self):
        while not self.stopping:
            now = time.monotonic()
            self.capacity(now)
            self.update_capacity(now)
            # Bound CPU time, not W size or execution concurrency. Admission
            # and HTTP feedback share this event loop; a large queue must not
            # monopolize it until every request has been handed off.
            admission_started = time.monotonic()
            admission_count = 0
            yielded = False
            while True:
                # Both routes borrow shared resources. Refresh all restoration
                # obligations atomically before EACH new admission; a local
                # flag alone lets text bypass an audio recovery (or vice versa).
                now = time.monotonic()
                if hasattr(self, 'coordinator'):
                    recovering = self.coordinator.admission_recovery_pending(now)
                else:
                    self.process_pause_leases(now)
                    recovering = self.recovery_pending
                if recovering:
                    break
                # W is a rate budget, not the native execution batch.
                # Returning KV owners may raise |W| above max_num_seqs;
                # only the scheduler limits simultaneous execution.
                rid = self.w.admit_next(self.capacity(now), now, native=self.config["arm"] == "fcfs")
                if rid is None:
                    break
                self.event("native_queued" if self.w.requests[rid].state == 'native_queued' else "admitted",
                           request_id=rid, rate=self.w.requests[rid].rate)
                self.ready[rid].set()
                admission_count += 1
                if time.monotonic() - admission_started >= .01:
                    yielded = True
                    break
            self.publish()
            elapsed = time.monotonic() - admission_started
            if elapsed >= .1:
                self.event('gateway_slow_admission_turn', duration_s=elapsed,
                           admitted=admission_count, yielded=yielded)
            # Publish before yielding; recheck live recovery/capacity next
            # turn. A CPU time slice is not an admission quota or rate limit.
            await asyncio.sleep(0 if yielded else .025)

    async def contract(self, request):
        return web.json_response(dict(session=self.session, modality=self.modality,
                                      clock_domain=self.clock_domain, arm=self.config["arm"]))

    async def receive_feedback(self, request):
        try:
            raw = await request.json()
            rid = raw["request_id"]
            if rid not in self.w.requests or self.w.requests[rid].state == "finished":
                return web.json_response({"error":"request not registered"}, status=409)
            f = ClientFeedback.parse(raw, now=time.monotonic(), session=self.session,
                    modality=self.modality, clock_domain=self.clock_domain,
                    rate=self.w.requests[rid].rate, previous=self.feedback.get(rid))
            if f.observed_at < self.registered_at[rid]:
                raise ValueError("feedback predates request registration")
            self.feedback[rid] = f
            self.observe_client_delivery(f)
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
            benchmark_rank = payload["omni_scheduling"].get("ingress_order")
            if self.ordered_backend:
                if type(benchmark_rank) is not int or benchmark_rank < 0:
                    raise ValueError("invalid benchmark ingress rank")
                await self.ordered_backend.register(benchmark_rank)
            self.w.add(rid, rate)
            self.registered_at[rid] = time.monotonic()
            self.ready[rid] = asyncio.Event()
            # Arrival order is server-owned, never supplied for priority gaming.
            payload["omni_scheduling"]["ingress_order"] = self.w.requests[rid].arrival_order
            self.event("arrived", request_id=rid, rate=rate, payload=payload)
            if self.engine_queue_enabled():
                # Registration must reach the target stage before this POST.
                # Upstream preparation never supplies client buffer credit.
                self.publish()
        except (ValueError, KeyError, TypeError) as e:
            return web.json_response({"error":str(e)}, status=400)
        response = None
        status = "cancelled"
        try:
            while not self.engine_queue_enabled() and not self.ready[rid].is_set():
                if request.transport is None or request.transport.is_closing():
                    raise ConnectionResetError("client disconnected while queued")
                try:
                    await asyncio.wait_for(self.ready[rid].wait(), .1)
                except asyncio.TimeoutError:
                    pass
            decoder = StreamDecoder(self.modality, self.special)
            self.event('upstream_forwarded', request_id=rid,
                       admission_scope='target_output_stage' if self.engine_queue_enabled() else 'http_ingress',
                       quota_state=self.w.requests[rid].state)
            upstream_context = (self.ordered_backend.post(self.http, self.config["upstream"] + "/v1/chat/completions", payload=payload, rid=rid, rank=benchmark_rank) if self.ordered_backend else self.http.post(self.config["upstream"] + "/v1/chat/completions", json=payload))
            async with upstream_context as upstream:
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
            self.finish_request(rid)
            self.event("terminal", request_id=rid, status=status)
            self.publish()

    def finish_request(self, rid):
        """Retire all control state after generation or transport termination."""
        self.w.complete(rid)
        for states in (self.feedback, self.delivery_history, self.ready,
                       self.registered_at, self.pause_acks, self.accepted_pause_leases):
            states.pop(rid, None)
        coordinator = getattr(self, 'coordinator', None)
        if coordinator is not None:
            coordinator.phase_attempts.pop(rid, None)

    async def startup(self, app):
        self.http = ClientSession(timeout=ClientTimeout(total=self.config.get("upstream_timeout_s", 600)),
                                  # Each upstream POST is a long-lived generation stream. Avoid
                                  # reusing an idle connection closed by the server while
                                  # admission queued the next request; do not retry POSTs
                                  # whose acceptance is uncertain.
                                  connector=__import__("aiohttp").TCPConnector(limit=0, force_close=True))
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
