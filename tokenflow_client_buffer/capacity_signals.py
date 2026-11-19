"""Execution evidence for capacity windows; never a source of client credit.

NVML is sampled at 0.5 s, not for each scheduler step. Complete output windows
provide rate lower bounds. Downward revisions additionally require independently
observed busy hardware doing scheduled work, or token-budget pressure with
runnable backlog. A gateway queue, policy holds and max_num_seqs alone prove
no such pressure.
"""
from __future__ import annotations

import math
import os


class CapacityExecutionSignals:
    def __init__(self, scheduler, config, *, samplers=None):
        self.scheduler = scheduler
        self.interval = float(config.get('capacity_signal_interval_s', .5))
        self.aggregation = config.get('capacity_gpu_aggregation', 'mean')
        self.samplers = samplers
        self.unavailable = None
        if samplers is None:
            try:
                from vllm_omni.scheduling.signals import NVMLGPUActiveSampler, parse_visible_device_ids
                ids = config.get('capacity_device_ids')
                if ids is None:
                    ids = parse_visible_device_ids(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
                self.samplers = [NVMLGPUActiveSampler(device_ids=[device], min_interval_s=self.interval)
                                 for device in ids]
            except (ImportError, ValueError) as error:
                self.samplers = []
                self.unavailable = type(error).__name__
        self.counters = dict(capacity_observed_s=0., capacity_hardware_valid_s=0.,
            capacity_hardware_busy_s=0., capacity_busy_work_s=0., capacity_backend_pressure_s=0.,
            capacity_budget_pressure_s=0., capacity_context_tokens=0.,
            capacity_context_requests=0, capacity_prefill_tokens=0,
            capacity_scheduled_requests=0)
        self.last_at = None
        self.last_sample = float('-inf')
        self.previous = dict(gpu=None, backlog=False, budget=False, work=False)
        self.latest = {}

    @staticmethod
    def _runnable(request):
        name = getattr(getattr(request, 'status', None), 'name', '').upper()
        return (not name or name in {'RUNNING', 'WAITING', 'PREEMPTED'})

    def record(self, output, now, *, paused_ids=()):
        scheduled = output.num_scheduled_tokens
        tokens = sum(scheduled.values())
        budget = getattr(self.scheduler, 'max_num_scheduled_tokens', 0)
        budget_full = bool(budget and tokens >= .95 * budget)
        if self.last_at is not None:
            dt = max(0., now - self.last_at)
            self.counters['capacity_observed_s'] += dt
            # Do not extend an old hardware sample across engine-idle gaps.
            fresh = 0 <= now - self.last_sample <= self.interval * 3
            if fresh and self.previous['gpu'] is not None:
                self.counters['capacity_hardware_valid_s'] += dt
                busy = self.previous['gpu'] >= .85
                self.counters['capacity_hardware_busy_s'] += dt * busy
                self.counters['capacity_busy_work_s'] += dt * (busy and self.previous['work'])
                self.counters['capacity_backend_pressure_s'] += dt * (busy and self.previous['backlog'])
            if fresh:
                self.counters['capacity_budget_pressure_s'] += dt * (self.previous['budget'] and self.previous['backlog'])
        self.last_at = now
        if now - self.last_sample >= self.interval:
            values = []
            unavailable = []
            for sampler in self.samplers:
                sample = sampler.sample(captured_monotonic_s=now)
                if sample.ratio is not None and (sample.sample_age_ms or 0) <= self.interval * 1500:
                    values.append(sample.ratio)
                else:
                    unavailable.append(sample.unavailable_reason)
            # A missing TP rank/device invalidates aggregate hardware evidence.
            gpu = None
            if values and len(values) == len(self.samplers):
                gpu = max(values) if self.aggregation == 'max' else sum(values)/len(values)
            requests = self.scheduler.requests
            waiting = list(getattr(self.scheduler, 'waiting', ()))
            running = list(getattr(self.scheduler, 'running', ()))
            paused = set(paused_ids)
            backlog = sum(self._runnable(r) and r.request_id not in paused
                          and r.request_id not in scheduled for r in waiting + running)
            contexts = []
            prefill = 0
            for rid, count in scheduled.items():
                request = requests.get(rid)
                if request is None:
                    continue
                before = max(0, request.num_computed_tokens - count)
                contexts.append(before)
                prefill += max(0, min(count, request.num_prompt_tokens - before))
            self.counters['capacity_context_tokens'] += sum(contexts)
            self.counters['capacity_context_requests'] += len(contexts)
            self.counters['capacity_prefill_tokens'] += prefill
            self.counters['capacity_scheduled_requests'] += len(scheduled)
            self.previous = dict(gpu=gpu, backlog=bool(backlog), budget=budget_full, work=bool(tokens))
            self.last_sample = now
            self.latest = dict(capacity_gpu_active_ratio=gpu,
                capacity_gpu_device_count=len(self.samplers),
                capacity_gpu_aggregation=self.aggregation,
                capacity_gpu_unavailable=unavailable or self.unavailable,
                capacity_runnable_backend_waiters=backlog,
                capacity_mean_context_tokens=sum(contexts)/len(contexts) if contexts else None,
                capacity_sampled_at=now)
        else:
            self.previous['budget'] = budget_full
            self.previous['work'] = bool(tokens)
        return {**self.counters, **self.latest}


def execution_window_evidence(current, previous, duration):
    """Return replayable pressure evidence, independent of admission decisions."""
    def delta(key):
        value = current.get(key, 0.) - previous.get(key, 0.)
        return max(0., value) if math.isfinite(value) else 0.
    covered = delta('capacity_observed_s')
    hardware = delta('capacity_hardware_valid_s')
    pressure = delta('capacity_backend_pressure_s')
    busy_work = delta('capacity_busy_work_s')
    token_pressure = delta('capacity_budget_pressure_s')
    context_n = delta('capacity_context_requests')
    valid = duration > 0 and covered >= .8 * duration
    hardware_saturated = valid and hardware >= .8 * duration and pressure >= .8 * duration
    hardware_service = valid and hardware >= .8 * duration and busy_work >= .8 * duration
    budget_saturated = valid and token_pressure >= .8 * duration
    saturated = hardware_saturated or hardware_service or budget_saturated
    return dict(saturated=saturated,
        saturation_basis=('hardware_busy_with_runnable_backlog' if hardware_saturated else
                          'hardware_busy_with_scheduled_work' if hardware_service else
                          'token_budget_with_runnable_backlog' if budget_saturated else 'underfilled'),
        hardware_coverage_ratio=hardware/duration if duration else 0.,
        hardware_pressure_ratio=pressure/duration if duration else 0.,
        hardware_busy_work_ratio=busy_work/duration if duration else 0.,
        token_budget_pressure_ratio=token_pressure/duration if duration else 0.,
        mean_context_tokens=delta('capacity_context_tokens')/context_n if context_n else None,
        sampled_context_requests=context_n,
        sampled_prefill_tokens=delta('capacity_prefill_tokens'),
        gpu_active_ratio=current.get('capacity_gpu_active_ratio'),
        gpu_device_count=current.get('capacity_gpu_device_count'),
        gpu_unavailable=current.get('capacity_gpu_unavailable'))
