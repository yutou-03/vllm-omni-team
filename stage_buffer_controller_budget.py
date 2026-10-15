"""Online stage-local inventory budget, using receiver-ready units.

Stage0 counts verified conditioning rows minus unique selected positions.
Stage1 counts new codec frames minus frames whose conversion has started.
The published count is retained as a cross-process consistency ceiling.
"""
from collections import defaultdict, deque
from pathlib import Path
import hashlib
import json
import math
import os
import time

from vllm_omni.profiling.stage_queue_trace import emit_stage_event


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.
    return values[min(len(values)-1, math.ceil(p*len(values))-1)]


class Inventory:
    def __init__(self):
        self.rows = 0
        self.received_rows = 0
        self.prompt = None
        self.text_chunks = set()
        self.text_received_chunks = set()
        self.selected = set()
        self.codec_sent = {}
        self.codec_received = {}
        self.codec_started = {}
        self.conversions = {}
        self.rates = {0: deque(maxlen=64), 1: deque(maxlen=64)}
        self.last_selection = None
        self.invalid = False
        self.ended_text = False

    def q(self, stage):
        if stage == 0:
            return None if self.prompt is None else self.rows-self.prompt-len(self.selected)
        return sum(self.codec_sent.values())-sum(self.codec_started.values())

    def q_ready(self, stage):
        if stage == 0:
            if self.prompt is None or 0 not in self.text_received_chunks:
                return None
            return self.received_rows-self.prompt-len(self.selected)
        return sum(self.codec_received.values())-sum(self.codec_started.values())

    def update(self, e):
        name = e['event']
        t = e['monotonic']
        if name == 'stage_study_text_payload_sent':
            cid = e['chunk_id']
            if cid in self.text_chunks:
                return
            if cid != len(self.text_chunks) or not e.get('raw_entity_embedding_layout_valid'):
                self.invalid = True
                return
            self.text_chunks.add(cid)
            self.rows += e['raw_entity_embedding_rows_total']
            prompt = e.get('raw_ids_prompt_count')
            if prompt is not None:
                if self.prompt is not None and self.prompt != prompt:
                    self.invalid = True
                self.prompt = prompt
        elif name == 'stage_study_text_payload_received':
            cid = e['chunk_id']
            if cid in self.text_received_chunks:
                return
            if not e.get('raw_entity_embedding_layout_valid'):
                self.invalid = True
                return
            self.text_received_chunks.add(cid)
            self.received_rows += e['raw_entity_embedding_rows_total']
            prompt = e.get('raw_ids_prompt_count')
            if prompt is not None:
                if self.prompt is not None and self.prompt != prompt:
                    self.invalid = True
                self.prompt = prompt
        elif name == 'stage_study_text_conditioning_selected':
            if e.get('conditioning_kind') in ('eos', 'pad'):
                self.ended_text = True
            if e.get('conditioning_kind') != 'output_embedding':
                return
            before = self.q(0)
            old = len(self.selected)
            self.selected.update(range(e['start_index'], e['end_index']))
            delta = len(self.selected)-old
            if delta and e.get('phase') == 'decode':
                if self.last_selection:
                    old_t, old_q = self.last_selection
                    dt = t-old_t
                    # Exclude immediate duplicate/bootstrap selection and input starvation.
                    if dt >= .005 and old_q is not None and old_q > 0:
                        self.rates[0].append(delta/dt)
                self.last_selection = (t, self.q(0))
        elif name == 'stage_study_chunk_sent':
            self.codec_sent.setdefault(e['chunk_id'], e['new_frames'])
        elif name == 'stage_study_chunk_received':
            self.codec_received.setdefault(e['chunk_id'], e['new_frames'])
        elif name == 'stage_study_chunk_conversion_started':
            self.codec_started.setdefault(e['chunk_id'], e['new_frames'])
            self.conversions.setdefault(e['chunk_id'], (t, e['new_frames'], e.get('stream_finished', False)))
        elif name == 'stage_study_chunk_conversion_completed':
            start = self.conversions.get(e['chunk_id'])
            if start and not start[2] and t > start[0]:
                self.rates[1].append(start[1]/(t-start[0]))


class Controller:
    def __init__(self, scheduler):
        self.stage = int(scheduler._omni_stage_id_for_trace())
        self.root = Path(os.environ['STAGE_QUEUE_TRACE_DIR']).parent
        config_override = os.environ.get('VLLM_OMNI_STAGE_BUFFER_CONFIG')
        config_path = (Path(config_override) if config_override else
            Path(__file__).resolve().parent/'experiments/stage_buffer_budget/controller_budget_config.json')
        self.config = json.loads(config_path.read_text())
        self.c = self.config['stages'][str(self.stage)]
        self.offset = 0
        self.state = defaultdict(Inventory)
        self.last_call = None
        self.last_had_running = False
        self.ticks = deque(maxlen=64)
        self.last_end = {}
        self.last_log = {}
        self.sequence = 0
        self.reasons = defaultdict(int)
        self.last_heartbeat = 0.
        self.watermark = 0.
        emit_stage_event('stage_buffer_policy_initialized', stage_id=self.stage,
            controller='stage_buffer_controller_budget:decide', policy=self.config,
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            observation='receiver-ready inventory bounded by published inventory', client_playback_credit=False,
            kv_preserving_hold=True)

    def refresh(self):
        path = self.root/'traces/stage_events.jsonl'
        if not path.exists():
            return
        # Freeze byte limit: records appended during this read belong to next tick.
        limit = path.stat().st_size
        with path.open('rb') as f:
            f.seek(self.offset)
            while f.tell() < limit:
                start = f.tell()
                line = f.readline()
                if not line.endswith(b'\n') or f.tell() > limit:
                    self.offset = start
                    break
                self.offset = f.tell()
                e = json.loads(line)
                name = e['event']
                if name not in {
                    'stage_study_text_payload_sent','stage_study_text_payload_received',
                    'stage_study_text_conditioning_selected',
                    'stage_study_chunk_sent','stage_study_chunk_received',
                    'stage_study_chunk_conversion_started',
                    'stage_study_chunk_conversion_completed'}:
                    continue
                self.watermark = max(self.watermark, e['monotonic'])
                rid = str(e.get('logical_request_id') or e.get('request_id'))
                self.state[rid].update(e)

    def fields(self, rid, now, guard):
        inv = self.state[rid]
        published_q = inv.q(self.stage)
        ready_q = inv.q_ready(self.stage)
        q = (min(published_q, ready_q) if published_q is not None
             and ready_q is not None and min(published_q, ready_q) >= 0
             and not inv.invalid else None)
        rate = max(self.c['calibration_rate_units_s'], percentile(inv.rates[self.stage], .95))
        coverage = q/rate if q is not None else None
        low = max(self.c['resume_below_s'], guard+self.config['margin_s'])
        high = max(self.c['pause_above_s'], low+self.config['margin_s']+self.config['min_hold_s'])
        budget = max(0., coverage-low-self.config['margin_s']) if coverage is not None else None
        duration = min(self.config['max_hold_s'], budget) if budget is not None else None
        return dict(inventory_units=q, published_inventory_units=published_q,
                    receiver_ready_inventory_units=ready_q,
                    consumption_rate_units_s=rate, coverage_s=coverage,
                    budget_s=budget, requested_hold_s=duration,
                    low_s=low, high_s=high, resume_guard_s=guard,
                    rate_samples=len(inv.rates[self.stage]), invalid_inventory=inv.invalid,
                    text_conditioning_ended=inv.ended_text,
                    unit='conditioning_position' if self.stage == 0 else 'new_codec_frame',
                    consumed_boundary='model_input_selected' if self.stage == 0 else 'conversion_started',
                    observation_watermark_s=self.watermark, trace_read_offset=self.offset,
                    decision_monotonic_s=now)

    def record(self, rid, now, reason, fields, **extra):
        self.reasons[reason] += 1
        action = reason in ('pause_accepted', 'resume_requested')
        if not action and now-self.last_log.get(rid, 0.) < self.config['no_action_log_interval_s']:
            return
        self.last_log[rid] = now
        self.sequence += 1
        emit_stage_event('stage_buffer_decision', stage_id=self.stage, request_id=rid,
                        sequence=self.sequence, reason=reason, **fields, **extra)

    def __call__(self, scheduler, now):
        start = time.monotonic()
        probe = scheduler._small_hold_probe
        if self.last_call is not None and self.last_had_running and scheduler.running:
            self.ticks.append(now-self.last_call)
        self.last_call = now
        self.last_had_running = bool(scheduler.running)
        self.refresh()
        now = time.monotonic()
        guard = max(self.c['minimum_resume_guard_s'], percentile(self.ticks, .95))
        if probe.buffer_policy_suspended:
            for rid, hold in list(probe.holds.items()):
                if hold.released_monotonic_s is None:
                    probe.release_hold(rid, 'policy_suspended')
            return
        for rid, hold in list(probe.holds.items()):
            fields = self.fields(rid, now, guard)
            if hold.released_monotonic_s is not None:
                self.last_end[rid] = now
                continue
            low_inventory = fields['coverage_s'] is None or fields['coverage_s'] <= fields['low_s']
            if low_inventory or now >= hold.deadline_monotonic_s:
                cause = 'inventory_low_or_unknown' if low_inventory else 'slice_deadline'
                probe.release_hold(rid, cause)
                self.last_end[rid] = now
                self.record(rid, now, 'resume_requested', fields, resume_reason=cause)
        for request in list(scheduler.running):
            rid = request.request_id
            if rid in probe.holds:
                continue
            fields = self.fields(rid, now, guard)
            if not probe._at_decode_boundary(request):
                reason = 'not_decode_boundary'
            elif now-self.last_end.get(rid, 0.) < self.config['cooldown_s']:
                reason = 'cooldown'
            elif fields['coverage_s'] is None:
                reason = 'inventory_unavailable'
            elif fields['coverage_s'] < fields['high_s']:
                reason = 'insufficient_downstream_coverage'
            elif fields['requested_hold_s'] < self.config['min_hold_s']:
                reason = 'insufficient_hold_budget'
            else:
                accepted = scheduler.request_hold(rid, fields['requested_hold_s'],
                    'stage_buffer_coverage', fields)
                reason = 'pause_accepted' if accepted else 'primitive_rejected'
            self.record(rid, now, reason, fields)
        if now-self.last_heartbeat >= 1.:
            self.last_heartbeat = now
            emit_stage_event('stage_buffer_policy_heartbeat', stage_id=self.stage,
                running=len(scheduler.running), held=len(probe.holds),
                decision_reason_counts=dict(self.reasons),
                controller_elapsed_ms=1000*(time.monotonic()-start),
                online_trace_bytes_read=self.offset)


_controllers = {}

def decide(scheduler, now):
    stage = int(scheduler._omni_stage_id_for_trace())
    if stage not in _controllers:
        _controllers[stage] = Controller(scheduler)
    _controllers[stage](scheduler, now)
