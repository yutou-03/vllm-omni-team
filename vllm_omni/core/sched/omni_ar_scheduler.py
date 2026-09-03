from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from time import monotonic, time
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler as AsyncVLLMScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.profiling.nvtx import nvtx_mark, nvtx_range
from vllm_omni.profiling.stage_queue_trace import emit_stage_event, stage_trace_span
from vllm_omni.recovery import PreemptionRecoveryMode
from vllm_omni.scheduling.policy import request_scheduling_metadata

logger = init_logger(__name__)

_CONTROLLED_PREEMPTION_FILE_ENV = "VLLM_OMNI_CONTROLLED_PREEMPTION_FILE"
_SLO_ACTIVE_PREEMPTION_ENV = "VLLM_OMNI_SLO_ACTIVE_PREEMPTION"
_SLO_ACTIVE_STAGES_ENV = "VLLM_OMNI_SLO_ACTIVE_STAGES"
_SLO_ACTIVE_GUARD_MS_ENV = "VLLM_OMNI_SLO_ACTIVE_GUARD_MS"
_SLO_PRIORITY_TRACE_ENV = "VLLM_OMNI_SLO_PRIORITY_TRACE"
_PREEMPTION_RECOVERY_MODE_ENV = "VLLM_OMNI_PREEMPTION_RECOVERY_MODE"


def _logical_request_id(request_id: str) -> str:
    """Return the frontend request id shared by all physical executions."""
    return request_id.split("__execution", 1)[0]


def _pipeline_occurrence(request_id: str) -> int:
    """Return 0 for the initial pipeline and N for ``__executionN``."""
    marker = "__execution"
    if marker not in request_id:
        return 0
    try:
        return int(request_id.rsplit(marker, 1)[1])
    except ValueError:
        return -1


def _load_controlled_preemption_rules() -> list[dict[str, Any]]:
    path = os.environ.get(_CONTROLLED_PREEMPTION_FILE_ENV)
    if not path:
        return []
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1:
        raise ValueError("controlled preemption file must use schema_version=1")
    events = document.get("events")
    if not isinstance(events, list):
        raise ValueError("controlled preemption file must contain an events list")

    normalized = []
    required = {
        "rule_id",
        "victim_request_id",
        "target_stage",
        "target_num_computed_tokens",
        "occurrence_index",
    }
    for index, event in enumerate(events):
        if not isinstance(event, dict) or not required.issubset(event):
            raise ValueError(f"controlled preemption event {index} is incomplete")
        rule = dict(event)
        rule["rule_id"] = str(rule["rule_id"])
        rule["victim_request_id"] = str(rule["victim_request_id"])
        rule["target_stage"] = int(rule["target_stage"])
        rule["target_num_computed_tokens"] = int(
            rule["target_num_computed_tokens"]
        )
        rule["occurrence_index"] = int(rule["occurrence_index"])
        if rule["target_stage"] < 0:
            raise ValueError(f"controlled preemption event {index} has negative stage")
        if rule["target_num_computed_tokens"] < 0:
            raise ValueError(f"controlled preemption event {index} has negative progress")
        if rule["occurrence_index"] < 0:
            raise ValueError(f"controlled preemption event {index} has negative occurrence")
        normalized.append(rule)
    rule_ids = [rule["rule_id"] for rule in normalized]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("controlled preemption rule_id values must be unique")
    return normalized


def _load_slo_active_preemption_config() -> tuple[bool, set[int], float, bool]:
    """Load the opt-in SLO active-switch experiment configuration."""
    raw_enabled = os.environ.get(_SLO_ACTIVE_PREEMPTION_ENV, "0").strip().lower()
    if raw_enabled not in {"0", "1", "false", "true"}:
        raise ValueError(
            f"{_SLO_ACTIVE_PREEMPTION_ENV} must be one of 0/1/false/true"
        )
    enabled = raw_enabled in {"1", "true"}
    raw_stages = os.environ.get(_SLO_ACTIVE_STAGES_ENV, "1")
    try:
        stages = {
            int(value.strip())
            for value in raw_stages.split(",")
            if value.strip()
        }
    except ValueError as error:
        raise ValueError(
            f"{_SLO_ACTIVE_STAGES_ENV} must be a comma-separated integer list"
        ) from error
    if not stages or any(stage_id < 0 for stage_id in stages):
        raise ValueError(f"{_SLO_ACTIVE_STAGES_ENV} must contain non-negative stages")
    try:
        guard_ms = float(os.environ.get(_SLO_ACTIVE_GUARD_MS_ENV, "0"))
    except ValueError as error:
        raise ValueError(f"{_SLO_ACTIVE_GUARD_MS_ENV} must be numeric") from error
    if not math.isfinite(guard_ms) or guard_ms < 0:
        raise ValueError(f"{_SLO_ACTIVE_GUARD_MS_ENV} must be finite and non-negative")
    raw_trace = os.environ.get(_SLO_PRIORITY_TRACE_ENV, "0").strip().lower()
    if raw_trace not in {"0", "1", "false", "true"}:
        raise ValueError(f"{_SLO_PRIORITY_TRACE_ENV} must be one of 0/1/false/true")
    trace_enabled = raw_trace in {"1", "true"}
    return enabled, stages, guard_ms, trace_enabled


def _load_preemption_recovery_mode() -> PreemptionRecoveryMode:
    """Load the recovery policy, preserving full restart as the default."""

    raw_mode = os.environ.get(
        _PREEMPTION_RECOVERY_MODE_ENV,
        PreemptionRecoveryMode.FULL_PIPELINE_RESTART.value,
    ).strip().lower()
    try:
        return PreemptionRecoveryMode(raw_mode)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in PreemptionRecoveryMode)
        raise ValueError(
            f"{_PREEMPTION_RECOVERY_MODE_ENV} must be one of: {choices}"
        ) from error


@dataclass
class KVCacheTransferData:
    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OmniARScheduler(OmniSchedulerMixin, VLLMScheduler):
    """Synchronous AutoRegressive scheduler for vLLM-Omni. This class is also
    used as a base class for the OmniARAsyncScheduler and holds most of the
    core scheduling logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track requests that need KV cache transfer when finished
        # Value is {"seq_len": int, "block_ids": list[int]}
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}

        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

        # Track ACTIVE transfers (submitted to runner but not yet acked via kv_extracted_req_ids)
        self.active_kv_transfers: set[str] = set()

        # Requests marked for deferred stop: keep running until KV extraction
        # completes so that kv_ready can be emitted while the request is still
        # alive.  Stopped on the first scheduler step after extraction ack.
        self.pending_stop_after_extraction: set[str] = set()

        # [Omni] Pre-parse KV transfer criteria
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()

        # Track requests that have already triggered prefill transfer to avoid duplicates
        self.transfer_triggered_requests: set[str] = set()

        # Cache per-request flag to avoid repeated deserialization of additional_information
        self._omits_kv_transfer_cache: dict[str, bool] = {}
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)
        # Snapshot prompt length for each streaming input update
        self._new_prompt_len_snapshot: dict[str, int] = {}
        self._initialize_baseline_scheduling()
        self._controlled_preemption_rules = [
            rule
            for rule in _load_controlled_preemption_rules()
            if rule["target_stage"] == int(self._omni_stage_id_for_trace())
        ]
        self._controlled_preemption_fired_rule_ids: set[str] = set()
        self._omni_preemption_causes_pending: dict[str, str] = {}
        self._omni_preemption_progress_pending: dict[str, int] = {}
        self._omni_preemption_rule_ids_pending: dict[str, str] = {}
        self._omni_preemption_recovery_modes_pending: dict[str, str] = {}
        self._preemption_recovery_mode = _load_preemption_recovery_mode()
        (
            self._slo_active_preemption_enabled,
            self._slo_active_preemption_stages,
            self._slo_active_guard_ms,
            self._slo_priority_trace_enabled,
        ) = _load_slo_active_preemption_config()
        self._slo_priority_witness_seen: set[tuple[str, str, int]] = set()

    def _get_confirmed_num_computed_tokens(self, request: Request) -> int:
        """num_computed_tokens minus async placeholders (KV actually on GPU)."""
        # Output placeholders are zero when async scheduling isn't used
        return request.num_computed_tokens - request.num_output_placeholders

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Apply the configured safe recovery action after resource preemption."""
        discarded_output_tokens = len(request._output_token_ids)
        discarded_computed_tokens = request.num_computed_tokens
        preemption_cause = str(
            getattr(request, "_omni_next_preemption_cause", "allocation_failure")
        )
        rule_id = getattr(request, "_omni_next_preemption_rule_id", None)
        if hasattr(request, "_omni_next_preemption_cause"):
            del request._omni_next_preemption_cause
        if hasattr(request, "_omni_next_preemption_rule_id"):
            del request._omni_next_preemption_rule_id
        old_output_connector_req_id = (
            getattr(request, "output_connector_req_id", None)
            or getattr(request, "connector_req_id", None)
            or getattr(request, "external_req_id", None)
            or request.request_id
        )

        super()._preempt_request(request, timestamp)

        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        configured_mode = getattr(
            self,
            "_preemption_recovery_mode",
            PreemptionRecoveryMode.FULL_PIPELINE_RESTART,
        )
        restart_talker_locally = (
            configured_mode is PreemptionRecoveryMode.STAGE_LOCAL_RECOMPUTE
            and getattr(model_config, "model_stage", None) == "talker"
        )
        actual_mode = (
            PreemptionRecoveryMode.STAGE_LOCAL_RECOMPUTE
            if restart_talker_locally
            else PreemptionRecoveryMode.FULL_PIPELINE_RESTART
        )
        recovery_modes = getattr(
            self, "_omni_preemption_recovery_modes_pending", None
        )
        if recovery_modes is not None:
            recovery_modes[request.request_id] = actual_mode.value

        if restart_talker_locally:
            del request._all_token_ids[request.num_prompt_tokens :]
            request._output_token_ids.clear()
            request.num_output_placeholders = 0
            request.discard_latest_async_tokens = False
            request._omni_execution_attempt = (
                getattr(request, "_omni_execution_attempt", 0) + 1
            )
            request.stop_reason = None
            request.output_connector_req_id = (
                f"{request.request_id}__stage{self._omni_stage_id_for_trace()}"
                f"__attempt{request.num_preemptions}"
            )
            self._omni_preemption_causes_pending[request.request_id] = (
                preemption_cause
            )
            self._omni_preemption_progress_pending[request.request_id] = (
                discarded_computed_tokens
            )
            if rule_id is not None:
                self._omni_preemption_rule_ids_pending[request.request_id] = str(
                    rule_id
                )
            if self.chunk_transfer_adapter is not None:
                self.chunk_transfer_adapter.abandon_sender(
                    str(old_output_connector_req_id)
                )
            emit_stage_event(
                "stage_local_dynamic_state_discarded",
                stage_id=self._omni_stage_id_for_trace(),
                request_id=request.request_id,
                discarded_output_tokens=discarded_output_tokens,
                discarded_computed_tokens=discarded_computed_tokens,
                retained_prompt_tokens=request.num_prompt_tokens,
                connector_req_id=request.output_connector_req_id,
                preemption_cause=preemption_cause,
                controlled_rule_id=rule_id,
                recovery_mode=actual_mode.value,
            )
            return

        # The upstream implementation immediately prepends the victim to the
        # waiting queue.  Full-pipeline recovery must never let that retired
        # physical request resume locally while the orchestrator is creating
        # its replacement execution.
        self.waiting.remove_requests([request])

        # An asynchronous GPU step launched before preemption may return after
        # the scheduler event is published.  Tag it stale immediately; the
        # orchestrator will abort this entire physical execution and submit a
        # new execution id from Stage 0.
        request._omni_execution_attempt = (
            getattr(request, "_omni_execution_attempt", 0) + 1
        )
        self._omni_preemption_causes_pending[request.request_id] = preemption_cause
        self._omni_preemption_progress_pending[request.request_id] = (
            discarded_computed_tokens
        )
        if rule_id is not None:
            self._omni_preemption_rule_ids_pending[request.request_id] = str(rule_id)
        if self.chunk_transfer_adapter is not None:
            self.chunk_transfer_adapter.abandon_sender(
                str(old_output_connector_req_id)
            )
        emit_stage_event(
            "full_pipeline_restart_requested",
            stage_id=self._omni_stage_id_for_trace(),
            request_id=request.request_id,
            discarded_output_tokens=discarded_output_tokens,
            discarded_computed_tokens=discarded_computed_tokens,
            connector_req_id=old_output_connector_req_id,
            preemption_cause=preemption_cause,
            controlled_rule_id=rule_id,
        )

    def _inject_controlled_preemptions(self, timestamp: float) -> set[str]:
        """Preempt pre-registered victims once their absolute progress is met."""
        if not self._controlled_preemption_rules:
            return set()

        injected: set[str] = set()
        running_by_logical_id = {
            _logical_request_id(str(request.request_id)): request
            for request in list(self.running)
        }
        for rule in self._controlled_preemption_rules:
            rule_id = rule["rule_id"]
            if rule_id in self._controlled_preemption_fired_rule_ids:
                continue
            request = running_by_logical_id.get(rule["victim_request_id"])
            if request is None:
                continue
            physical_request_id = str(request.request_id)
            if _pipeline_occurrence(physical_request_id) != rule["occurrence_index"]:
                continue
            progress = self._get_confirmed_num_computed_tokens(request)
            if progress < rule["target_num_computed_tokens"]:
                continue

            self.running.remove(request)
            request._omni_next_preemption_cause = "controlled_restart"
            request._omni_next_preemption_rule_id = rule_id
            self._preempt_request(request, timestamp)
            self._controlled_preemption_fired_rule_ids.add(rule_id)
            injected.add(physical_request_id)
            emit_stage_event(
                "controlled_preemption_injected",
                stage_id=self._omni_stage_id_for_trace(),
                request_id=physical_request_id,
                controlled_rule_id=rule_id,
                victim_request_id=rule["victim_request_id"],
                occurrence_index=rule["occurrence_index"],
                target_num_computed_tokens=rule["target_num_computed_tokens"],
                actual_num_computed_tokens=progress,
                injection_cause="controlled_restart",
                injection_succeeded=True,
            )
        return injected

    def _slo_stage_fields(self, request: Request) -> tuple[float, float]:
        """Return the absolute Stage deadline and predicted Stage service ms."""
        stage_id = int(self._omni_stage_id_for_trace())
        metadata = request_scheduling_metadata(request)
        deadlines = metadata.get("sched_stage_deadline_monotonic_s")
        predictions = metadata.get("sched_predicted_stage_ms")
        if not isinstance(deadlines, (list, tuple)) or stage_id >= len(deadlines):
            raise ValueError(
                f"request {request.request_id!r} has no Stage-{stage_id} deadline"
            )
        if not isinstance(predictions, (list, tuple)) or stage_id >= len(predictions):
            raise ValueError(
                f"request {request.request_id!r} has no Stage-{stage_id} prediction"
            )
        deadline = float(deadlines[stage_id])
        predicted_ms = float(predictions[stage_id])
        if not math.isfinite(deadline) or not math.isfinite(predicted_ms):
            raise ValueError(
                f"request {request.request_id!r} has non-finite SLO metadata"
            )
        if predicted_ms <= 0:
            raise ValueError(
                f"request {request.request_id!r} has non-positive Stage prediction"
            )
        return deadline, predicted_ms

    def _maybe_preempt_for_slo(
        self,
        timestamp: float,
        *,
        monotonic_now: float | None = None,
    ) -> set[str]:
        """Actively replace one running request when an earlier SLO is urgent.

        The passive and active experiment groups use identical Stage-deadline
        ordering.  This opt-in hook changes only the running-to-waiting
        boundary: with no free sequence slot, the earliest-deadline waiting
        request may replace the latest-deadline running request once its slack
        can no longer cover its own predicted Stage service time.
        """
        stage_id = int(self._omni_stage_id_for_trace())
        if not (
            getattr(self, "_slo_active_preemption_enabled", False)
            or getattr(self, "_slo_priority_trace_enabled", False)
        ):
            return set()
        if stage_id not in getattr(self, "_slo_active_preemption_stages", {1}):
            return set()
        waiting = list(self.waiting)
        running = list(self.running)
        max_running = int(getattr(self, "max_num_running_reqs", len(running)))
        if not waiting or not running or len(running) < max_running:
            return set()

        waiting_with_fields = [
            (request, *self._slo_stage_fields(request)) for request in waiting
        ]
        running_with_fields = [
            (request, *self._slo_stage_fields(request)) for request in running
        ]
        aggressor, aggressor_deadline, aggressor_predicted_ms = min(
            waiting_with_fields,
            key=lambda item: (item[1], str(item[0].request_id)),
        )
        victim, victim_deadline, _ = max(
            running_with_fields,
            key=lambda item: (item[1], str(item[0].request_id)),
        )
        if aggressor_deadline >= victim_deadline:
            return set()

        if monotonic_now is None:
            monotonic_now = timestamp
        slack_ms = (aggressor_deadline - monotonic_now) * 1000.0
        trigger_budget_ms = aggressor_predicted_ms + float(
            getattr(self, "_slo_active_guard_ms", 0.0)
        )
        witness_key = (
            _logical_request_id(str(aggressor.request_id)),
            _logical_request_id(str(victim.request_id)),
            stage_id,
        )
        if witness_key not in self._slo_priority_witness_seen:
            self._slo_priority_witness_seen.add(witness_key)
            emit_stage_event(
                "strict_priority_blocking_witness",
                stage_id=stage_id,
                aggressor_request_id=str(aggressor.request_id),
                victim_request_id=str(victim.request_id),
                aggressor_deadline_monotonic_s=aggressor_deadline,
                victim_deadline_monotonic_s=victim_deadline,
                aggressor_slack_ms=slack_ms,
                aggressor_predicted_stage_ms=aggressor_predicted_ms,
                trigger_budget_ms=trigger_budget_ms,
                active_preemption_enabled=bool(
                    getattr(self, "_slo_active_preemption_enabled", False)
                ),
                blocking_reason="sequence_slots_full",
            )

        if not getattr(self, "_slo_active_preemption_enabled", False):
            return set()
        if slack_ms > trigger_budget_ms:
            return set()

        physical_victim_id = str(victim.request_id)
        self.running.remove(victim)
        victim._omni_next_preemption_cause = "slo_active_switch"
        progress = self._get_confirmed_num_computed_tokens(victim)
        self._preempt_request(victim, timestamp)
        emit_stage_event(
            "slo_active_preemption",
            stage_id=stage_id,
            aggressor_request_id=str(aggressor.request_id),
            victim_request_id=physical_victim_id,
            victim_num_computed_tokens=progress,
            aggressor_deadline_monotonic_s=aggressor_deadline,
            victim_deadline_monotonic_s=victim_deadline,
            aggressor_slack_ms=slack_ms,
            aggressor_predicted_stage_ms=aggressor_predicted_ms,
            trigger_budget_ms=trigger_budget_ms,
            preemption_cause="slo_active_switch",
        )
        return {physical_victim_id}

    def _get_kv_transfer_criteria(self) -> dict | None:
        # Note: vllm_config is available in Scheduler after super().__init__
        if not hasattr(self, "vllm_config"):
            return None

        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            if isinstance(omni_kv_config, dict):
                return omni_kv_config.get("kv_transfer_criteria", None)
            else:
                return getattr(omni_kv_config, "kv_transfer_criteria", None)
        return None

    def _request_omits_kv_transfer_to_next_stage(self, request: Request) -> bool:
        """True when orchestrator will not run stage 1+ for this request (e.g. text-only).

        The result is cached per request to avoid repeated deserialization of
        additional_information on every scheduler tick.
        """
        rid = request.request_id
        cached = self._omits_kv_transfer_cache.get(rid)
        if cached is not None:
            return cached

        payload = getattr(request, "additional_information", None)
        if payload is None:
            result = False
        else:
            info = deserialize_additional_information(payload)
            result = info.get("omni_final_stage_id") == 0

        self._omits_kv_transfer_cache[rid] = result
        return result

    def _process_kv_transfer_trigger(self, request: Request, new_token_ids: list[int]) -> bool:
        """
        Check triggers and process side effects (marking transfer).
        Returns True if request should be STOPPED.
        Returns False if request should continue (even if transfer was triggered).
        """
        if not self.kv_transfer_criteria:
            return False

        # Text-only requests finalize at stage 0; do not prefill-stop for DiT KV.
        if self._request_omits_kv_transfer_to_next_stage(request):
            return False

        if request.request_id in self.waiting_for_transfer_free:
            return False

        criteria_type = self.kv_transfer_criteria.get("type")
        stop_decode_on_trigger = self.kv_transfer_criteria.get("stop_after_transfer", True)

        if request.request_id in self.transfer_triggered_requests:
            # Deferred stop: once KV extraction is complete (no longer in
            # active_kv_transfers), stop the request.  This guarantees the
            # kv_ready signal was emitted while the request was still alive.
            if (
                request.request_id in self.pending_stop_after_extraction
                and request.request_id not in self.active_kv_transfers
            ):
                self.pending_stop_after_extraction.discard(request.request_id)
                request.status = RequestStatus.FINISHED_STOPPED
                return True
            return False

        # seq_len for KV transfer must exclude async placeholders.
        confirmed_computed = self._get_confirmed_num_computed_tokens(request)

        if criteria_type == "prefill_finished":
            if confirmed_computed >= request.num_prompt_tokens:
                self.transfer_triggered_requests.add(request.request_id)
                self._mark_request_for_kv_transfer(request.request_id, confirmed_computed)
                actually_queued = request.request_id in self.requests_needing_kv_transfer

                if stop_decode_on_trigger and actually_queued:
                    # Defer the stop until KV extraction completes so that
                    # the kv_ready signal can be emitted while the request
                    # is still alive.  The request will be stopped on the
                    # next scheduler step after extraction ack arrives.
                    self.pending_stop_after_extraction.add(request.request_id)

                return False

        elif criteria_type == "special_token":
            target_token_id = self.kv_transfer_criteria.get("token_id")
            if target_token_id is not None and target_token_id in new_token_ids:
                self.transfer_triggered_requests.add(request.request_id)

                try:
                    idx = new_token_ids.index(target_token_id)
                    tokens_to_exclude = len(new_token_ids) - (idx + 1)
                    snapshot_len = confirmed_computed - tokens_to_exclude
                except ValueError:
                    snapshot_len = confirmed_computed

                self._mark_request_for_kv_transfer(request.request_id, snapshot_len)
                actually_queued = request.request_id in self.requests_needing_kv_transfer

                if stop_decode_on_trigger and actually_queued:
                    self.pending_stop_after_extraction.add(request.request_id)

                return False

        return False

    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        nvtx_mark("omni_ar:schedule_start")
        iteration_id = self._omni_next_iteration_id()
        stage_id = self._omni_stage_id_for_trace()
        schedule_start = time()
        num_running_before = len(self.running)
        num_waiting_before = len(self.waiting)
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)
            # Reset queue-time tracking for pre-warmed requests that just
            # received their first real chunk.  The pre-warm dummy request
            # fires a QUEUED event too early (before upstream data exists);
            # clearing events and re-recording QUEUED makes the subsequent
            # SCHEDULED meaningful — queue_time then measures wait from
            # real-data-ready until the scheduler admits the request.
            for req_id in self.chunk_transfer_adapter.requests_with_ready_chunks:
                request = self.requests.get(req_id)
                if request is None:
                    continue
                if getattr(request, "_omni_first_real_chunk_handled", False):
                    continue
                request.events.clear()
                request.record_event(EngineCoreEventType.QUEUED)
                request._omni_first_real_chunk_handled = True

        self._baseline_prepare_schedule()
        slo_active_preempted_req_ids = self._maybe_preempt_for_slo(
            schedule_start,
            monotonic_now=monotonic(),
        )
        controlled_preempted_req_ids = self._inject_controlled_preemptions(
            schedule_start
        )

        try:
            with (
                stage_trace_span(
                    "stage_schedule_base",
                    stage_id=stage_id,
                    nvtx_name=f"s{stage_id}_schedule_base",
                    nvtx_color="gray",
                    queue_len=num_waiting_before,
                    active_reqs=num_running_before,
                ),
                nvtx_range("omni:base_schedule"),
            ):
                scheduler_output = super().schedule()
            scheduler_output.preempted_req_ids = set(
                scheduler_output.preempted_req_ids
            ) | controlled_preempted_req_ids | slo_active_preempted_req_ids
            scheduler_output.stage_preemption_causes = {
                request_id: self._omni_preemption_causes_pending[request_id]
                for request_id in scheduler_output.preempted_req_ids
                if request_id in self._omni_preemption_causes_pending
            }
            scheduler_output.stage_preemption_progress = {
                request_id: self._omni_preemption_progress_pending[request_id]
                for request_id in scheduler_output.preempted_req_ids
                if request_id in self._omni_preemption_progress_pending
            }
            scheduler_output.stage_preemption_rule_ids = {
                request_id: self._omni_preemption_rule_ids_pending[request_id]
                for request_id in scheduler_output.preempted_req_ids
                if request_id in self._omni_preemption_rule_ids_pending
            }
            scheduler_output.stage_preemption_recovery_modes = {
                request_id: self._omni_preemption_recovery_modes_pending[request_id]
                for request_id in scheduler_output.preempted_req_ids
                if request_id in self._omni_preemption_recovery_modes_pending
            }
            for rid, ntok in scheduler_output.num_scheduled_tokens.items():
                nvtx_mark(f"scheduler_scheduled:req={str(rid)[-8:]}:ntok={ntok}")
            self._trace_scheduler_output(
                scheduler_output,
                iteration_id=iteration_id,
                timestamp_start=schedule_start,
                timestamp_end=time(),
                num_running_before=num_running_before,
                num_waiting_before=num_waiting_before,
            )
        finally:
            if self.chunk_transfer_adapter:
                # Add request waiting for chunk to the waiting and running queue
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    external_req_id=(getattr(request, "external_req_id", None) if request else None),
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output, self.requests)
            # Add information about requests needing KV cache transfer
            finished_reqs = self.get_finished_requests_needing_kv_transfer()
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
            finished_reqs = {}

        # Wrap in omni scheduler output to carry transfer metadata.
        base_fields = SchedulerOutput.__dataclass_fields__.keys()
        base_data = {name: getattr(scheduler_output, name) for name in base_fields}
        nvtx_mark("omni_ar:schedule_end")
        request_ids_with_connector_state = set(
            scheduler_output.num_scheduled_tokens
        ) | set(scheduler_output.preempted_req_ids)
        omni_output = OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_reqs,
            request_execution_attempts={
                request_id: getattr(self.requests.get(request_id), "_omni_execution_attempt", 0)
                for request_id in scheduler_output.num_scheduled_tokens
            },
            request_output_connector_req_ids={
                request_id: str(output_connector_req_id)
                for request_id in request_ids_with_connector_state
                if (request := self.requests.get(request_id)) is not None
                and (
                    output_connector_req_id := getattr(
                        request,
                        "output_connector_req_id",
                        None,
                    )
                )
                is not None
            },
            stage_preemption_causes=dict(
                getattr(scheduler_output, "stage_preemption_causes", {})
            ),
            stage_preemption_progress=dict(
                getattr(scheduler_output, "stage_preemption_progress", {})
            ),
            stage_preemption_rule_ids=dict(
                getattr(scheduler_output, "stage_preemption_rule_ids", {})
            ),
            stage_preemption_recovery_modes=dict(
                getattr(scheduler_output, "stage_preemption_recovery_modes", {})
            ),
        )
        for request_id in omni_output.preempted_req_ids:
            self._omni_preemption_causes_pending.pop(request_id, None)
            self._omni_preemption_progress_pending.pop(request_id, None)
            self._omni_preemption_rule_ids_pending.pop(request_id, None)
            self._omni_preemption_recovery_modes_pending.pop(request_id, None)
        return omni_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        stage_id = getattr(self.vllm_config.model_config, "stage_id", "?")
        with (
            stage_trace_span(
                "stage_update_from_output",
                stage_id=stage_id,
                nvtx_name=f"s{stage_id}_update_from_output",
                nvtx_color="teal",
                batch_num_tokens=int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0),
                batch_num_seqs=len(getattr(scheduler_output, "num_scheduled_tokens", {}) or {}),
            ),
            nvtx_range("omni_ar:update_from_output"),
        ):
            sampled_token_ids = model_runner_output.sampled_token_ids
            logprobs = model_runner_output.logprobs
            prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            pooler_outputs = model_runner_output.pooler_output
            num_nans_in_logits = model_runner_output.num_nans_in_logits
            kv_connector_output = model_runner_output.kv_connector_output
            cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats

            perf_stats: PerfStats | None = None
            if self.perf_metrics and self.perf_metrics.is_enabled():
                perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

            outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
            spec_decoding_stats: SpecDecodingStats | None = None
            kv_connector_stats: KVConnectorStats | None = (
                kv_connector_output.kv_connector_stats if kv_connector_output else None
            )
            if kv_connector_stats and self.connector:
                kv_stats = self.connector.get_kv_connector_stats()
                if kv_stats:
                    kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

            failed_kv_load_req_ids = None
            if kv_connector_output and kv_connector_output.invalid_block_ids:
                # These blocks contain externally computed tokens that failed to
                # load. Identify affected requests and adjust their computed token
                # count to trigger recomputation of the invalid blocks.
                failed_kv_load_req_ids = self._handle_invalid_blocks(
                    kv_connector_output.invalid_block_ids,
                    num_scheduled_tokens,
                )

            # Pre-process KV extraction acks so that the per-request loop below
            # can see up-to-date active_kv_transfers state and emit kv_ready
            # signals while requests are still alive (before any deferred stop).
            kv_extracted_ids = getattr(model_runner_output, "kv_extracted_req_ids", None)
            if kv_extracted_ids:
                for req_id in kv_extracted_ids:
                    try:
                        self.active_kv_transfers.discard(req_id)
                        req = self.requests.get(req_id)
                        if req is not None and not req.is_finished():
                            outputs[req.client_index].append(
                                EngineCoreOutput(
                                    request_id=req_id,
                                    new_token_ids=[],
                                    kv_transfer_params={"kv_ready": True},
                                )
                            )
                    except Exception:
                        init_logger(__name__).exception("Failed to pre-process KV extraction for %s", req_id)

            # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
            # the below loop can be a performance bottleneck. We should do our best
            # to avoid expensive operations inside the loop.
            stopped_running_reqs: set[Request] = set()
            stopped_preempted_reqs: set[Request] = set()
            for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
                assert num_tokens_scheduled > 0
                if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                    # Skip requests that were recovered from KV load failure
                    continue
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    # The request is already finished. This can happen if the
                    # request is aborted while the model is executing it (e.g.,
                    # in pipeline parallelism or async scheduling).
                    continue

                req_index = model_runner_output.req_id_to_index[req_id]
                generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

                scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
                if scheduled_spec_token_ids and generated_token_ids:
                    num_draft_tokens = len(scheduled_spec_token_ids)
                    num_accepted = len(generated_token_ids) - 1
                    num_rejected = num_draft_tokens - num_accepted
                    # num_computed_tokens represents the number of tokens
                    # processed in the current step, considering scheduled
                    # tokens and rejections. If some tokens are rejected,
                    # num_computed_tokens is decreased by the number of rejected
                    # tokens.
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    # If async scheduling, num_output_placeholders also includes
                    # the scheduled spec tokens count and so is similarly adjusted.
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                    spec_decoding_stats = self.make_spec_decoding_stats(
                        spec_decoding_stats,
                        num_draft_tokens=num_draft_tokens,
                        num_accepted_tokens=num_accepted,
                        num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                        request_id=req_id,
                    )

                # Free encoder inputs only after the step has actually executed.
                if request.has_encoder_inputs:
                    self._free_encoder_inputs(request)

                stopped = False
                is_segment_finished = False
                new_logprobs = None
                new_token_ids = generated_token_ids
                pooler_output = pooler_outputs[req_index] if pooler_outputs else None
                kv_transfer_params = None
                status_before_stop = request.status
                finish_reason = None
                routed_experts = None
                scheduled_attempt = getattr(
                    scheduler_output, "request_execution_attempts", {}
                ).get(req_id, 0)
                current_attempt = getattr(request, "_omni_execution_attempt", 0)
                discard_async_output = scheduled_attempt != current_attempt

                # Check for stop and update request status.
                if discard_async_output:
                    new_token_ids = []
                    pooler_output = None
                elif new_token_ids:
                    new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
                elif request.pooling_params and pooler_output is not None:
                    # Pooling stops as soon as there is output.
                    request.status = RequestStatus.FINISHED_STOPPED
                    stopped = True

                # Async scheduling may finish an already-launched model step
                # after this request was preempted.  The async scheduler drops
                # the sampled token above, but the Omni model runner also
                # returns a pooler payload used by the stage connector.  That
                # payload belongs to the abandoned execution attempt as well;
                # forwarding it under the new connector namespace would mix
                # old codec frames into the restarted downstream request.
                # If criteria returns True, it means we must STOP the request.
                # If criteria returns False, it might have triggered a background
                # transfer (e.g. prefill finished / special token) but continues decoding.
                if (
                    not discard_async_output
                    and not stopped
                    and self._process_kv_transfer_trigger(request, new_token_ids)
                ):
                    stopped = True

                if not discard_async_output:
                    self._trace_recovery_progress_after_output(
                        request,
                        num_tokens_scheduled=num_tokens_scheduled,
                    )

                if new_token_ids and self.structured_output_manager.should_advance(request):
                    struct_output_request = request.structured_output_request
                    assert struct_output_request is not None
                    assert struct_output_request.grammar is not None
                    if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
                        logger.error(
                            "Unexpected: grammar rejected tokens %s for request %s. Terminating request.",
                            new_token_ids,
                            req_id,
                        )
                        request.status = RequestStatus.FINISHED_ERROR
                        request.resumable = False
                        stopped = True

                if stopped:
                    routed_experts = self._get_routed_experts(request)

                    # Capture finish_reason BEFORE _handle_stopped_request, which may
                    # reset the status to WAITING for streaming requests that continue.
                    finish_reason = request.get_finished_reason()
                    is_segment_finished = request.is_finished() and request.resumable
                    finished = self._handle_stopped_request(request)
                    if finished:
                        kv_transfer_params = self._free_request(request)
                    if status_before_stop == RequestStatus.RUNNING:
                        stopped_running_reqs.add(request)
                    elif status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                        # In async chunk mode, request may be in either queue.
                        # Remove from both to avoid stale queue entries.
                        stopped_running_reqs.add(request)
                        stopped_preempted_reqs.add(request)
                    else:
                        stopped_preempted_reqs.add(request)

                # Extract sample logprobs if needed.
                if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

                if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                    request.num_nans_in_logits = num_nans_in_logits[req_id]

                # Get prompt logprobs for this request.
                prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
                if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:
                    is_first_s2_output = False
                    output_token_ids = getattr(request, "output_token_ids", None)
                    if output_token_ids is None:
                        output_token_ids = getattr(request, "_output_token_ids", [])
                    output_len = len(output_token_ids)
                    if new_token_ids:
                        for token_id in new_token_ids:
                            emit_stage_event(
                                "stage_token_output",
                                stage_id=self.vllm_config.model_config.stage_id,
                                request_id=req_id,
                                token_id=token_id,
                                num_new_tokens=len(new_token_ids),
                                output_len=output_len,
                            )
                        if not getattr(request, "_omni_stage_queue_first_output_emitted", False):
                            stage_id = self.vllm_config.model_config.stage_id
                            if str(stage_id) == "2":
                                with nvtx_range(
                                    f"TTFP:s2_first_output_ready:req={str(req_id)[-8:]}",
                                    color="orange",
                                ):
                                    nvtx_mark(
                                        f"TTFP:s2_first_output_ready:req={str(req_id)[-8:]}",
                                        color="orange",
                                    )
                                is_first_s2_output = True
                            emit_stage_event(
                                "stage_first_output",
                                stage_id=stage_id,
                                request_id=req_id,
                                output_len=output_len,
                            )
                            request._omni_stage_queue_first_output_emitted = True
                    if stopped:
                        emit_stage_event(
                            "stage_done",
                            stage_id=self.vllm_config.model_config.stage_id,
                            request_id=req_id,
                            output_len=output_len,
                            finish_reason=finish_reason,
                        )
                    # Add EngineCoreOutput for this Request.
                    output_context = (
                        nvtx_range(
                            f"TTFP:s2_engine_core_output_create:req={str(req_id)[-8:]}",
                            color="orange",
                        )
                        if is_first_s2_output
                        else nullcontext()
                    )
                    with output_context:
                        outputs[request.client_index].append(
                            EngineCoreOutput(
                                request_id=req_id,
                                new_token_ids=new_token_ids,
                                finish_reason=finish_reason,
                                new_logprobs=new_logprobs,
                                new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                                pooling_output=pooler_output,
                                stop_reason=request.stop_reason,
                                events=request.take_events(),
                                prefill_stats=request.take_prefill_stats(),
                                kv_transfer_params=kv_transfer_params,
                                trace_headers=request.trace_headers,
                                routed_experts=routed_experts,
                                num_nans_in_logits=request.num_nans_in_logits,
                                is_segment_finished=is_segment_finished,
                                new_prompt_len_snapshot=self._new_prompt_len_snapshot.get(req_id, None),
                            )
                        )
                    if is_first_s2_output:
                        nvtx_mark(
                            f"TTFP:s2_engine_core_output_created:req={str(req_id)[-8:]}",
                            color="orange",
                        )
                    if self.chunk_transfer_adapter is not None:
                        self.chunk_transfer_adapter.save_async(pooler_output, request)
                else:
                    # Invariant: EngineCore returns no partial prefill outputs.
                    assert not prompt_logprobs_tensors

            # Remove the stopped requests from the running and waiting queues.
            if stopped_running_reqs:
                self.running = remove_all(self.running, stopped_running_reqs)
            if stopped_preempted_reqs:
                # This is a rare case and unlikely to impact performance.
                self.waiting.remove_requests(stopped_preempted_reqs)
                self.skipped_waiting.remove_requests(stopped_preempted_reqs)

            # [Main] Handle failed KV load requests
            if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
                requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
                self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
                for request in requests:
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=request.get_finished_reason(),
                            events=request.take_events(),
                            trace_headers=request.trace_headers,
                        )
                    )
                    if self.chunk_transfer_adapter is not None:
                        self.chunk_transfer_adapter.cleanup_receiver(
                            request.request_id,
                        )

            # [Omni] Cleanup state for finished requests
            for req in stopped_running_reqs:
                if req.request_id not in self.waiting_for_transfer_free:
                    if req.request_id in self.transfer_triggered_requests:
                        self.transfer_triggered_requests.remove(req.request_id)
                    if req.request_id in self.active_kv_transfers:
                        self.active_kv_transfers.remove(req.request_id)
                    self.pending_stop_after_extraction.discard(req.request_id)

            # Same for preempted
            for req in stopped_preempted_reqs:
                if req.request_id not in self.waiting_for_transfer_free:
                    if req.request_id in self.transfer_triggered_requests:
                        self.transfer_triggered_requests.remove(req.request_id)
                    if req.request_id in self.active_kv_transfers:
                        self.active_kv_transfers.remove(req.request_id)
                    self.pending_stop_after_extraction.discard(req.request_id)

            # KV Connector: update state for finished KV Transfers.
            if kv_connector_output:
                self._update_from_kv_xfer_finished(kv_connector_output)

            # collect KV cache events from KV cache manager
            events = self.kv_cache_manager.take_events()

            # collect KV cache events from connector
            if self.connector is not None:
                connector_events = self.connector.take_events()
                if connector_events:
                    if events is None:
                        events = list(connector_events)
                    else:
                        events.extend(connector_events)

            # publish collected KV cache events
            if events:
                batch = KVEventBatch(ts=time(), events=events)
                self.kv_event_publisher.publish(batch)

            # Create EngineCoreOutputs for all clients that have requests with
            # outputs in this step.
            engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

            # A preemption can happen without producing an EngineCoreOutput for
            # the victim.  Publish it on the batch envelope so the multi-stage
            # orchestrator can invalidate downstream consumers and create a
            # new full-pipeline execution.
            preempted_by_client: dict[int, list[str]] = defaultdict(list)
            for req_id in getattr(scheduler_output, "preempted_req_ids", set()) or set():
                request = self.requests.get(req_id)
                client_index = request.client_index if request is not None else 0
                preempted_by_client[client_index].append(str(req_id))
            for client_index, request_ids in preempted_by_client.items():
                if (eco := engine_core_outputs.get(client_index)) is None:
                    engine_core_outputs[client_index] = eco = EngineCoreOutputs()
                eco.stage_preempted_request_ids = sorted(request_ids)
                eco.stage_preempted_connector_req_ids = {
                    req_id: str(
                        getattr(
                            self.requests.get(req_id),
                            "output_connector_req_id",
                            req_id,
                        )
                    )
                    for req_id in request_ids
                }
                eco.stage_preemption_causes = {
                    req_id: getattr(
                        scheduler_output, "stage_preemption_causes", {}
                    ).get(req_id, "allocation_failure")
                    for req_id in request_ids
                }
                eco.stage_preemption_progress = {
                    req_id: int(
                        getattr(
                            scheduler_output, "stage_preemption_progress", {}
                        ).get(req_id, 0)
                    )
                    for req_id in request_ids
                }
                eco.stage_preemption_rule_ids = {
                    req_id: rule_id
                    for req_id in request_ids
                    if (
                        rule_id := getattr(
                            scheduler_output, "stage_preemption_rule_ids", {}
                        ).get(req_id)
                    )
                    is not None
                }
                eco.stage_preemption_recovery_modes = {
                    req_id: getattr(
                        scheduler_output,
                        "stage_preemption_recovery_modes",
                        {},
                    ).get(
                        req_id,
                        PreemptionRecoveryMode.FULL_PIPELINE_RESTART.value,
                    )
                    for req_id in request_ids
                }

            finished_req_ids = self.finished_req_ids_dict
            if finished_req_ids:
                # Include ids of requests that finished since last outputs
                # were sent.
                for client_index, finished_set in finished_req_ids.items():
                    # Set finished request set in EngineCoreOutputs for this client.
                    if (eco := engine_core_outputs.get(client_index)) is not None:
                        eco.finished_requests = finished_set
                    else:
                        engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
                finished_req_ids.clear()

            if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
                # Return stats to only one of the front-ends.
                if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                    # We must return the stats even if there are no request
                    # outputs this step.
                    engine_core_outputs[0] = eco = EngineCoreOutputs()
                eco.scheduler_stats = stats

            # Free blocks that were held for transfer (kv_ready and
            # active_kv_transfers updates already done before the per-request loop).
            if kv_extracted_ids:
                for req_id in kv_extracted_ids:
                    try:
                        if req_id in self.waiting_for_transfer_free:
                            req = self.requests.get(req_id)
                            if req:
                                self.kv_cache_manager.free(req)
                                if req_id in self.requests:
                                    del self.requests[req_id]
                                if req_id in self.transfer_triggered_requests:
                                    self.transfer_triggered_requests.remove(req_id)
                                self.active_kv_transfers.discard(req_id)
                                self.pending_stop_after_extraction.discard(req_id)
                                logger.debug(f"Freed blocks for {req_id} after transfer extraction")
                            self.waiting_for_transfer_free.remove(req_id)
                    except Exception:
                        init_logger(__name__).exception("Failed to free blocks for %s after transfer", req_id)

        return engine_core_outputs

    def finish_requests(self, request_ids: Any, finished_status: RequestStatus) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """

        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.finish_requests(request_ids, finished_status, self.requests)

        result = super().finish_requests(request_ids, finished_status)
        self._baseline_forget_request_ids(request_ids)
        return result

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Override: Only extend prompt at stage 0, and replace
        the existing session with the next streaming update at other stages.

        Discards the last sampled output token from the prior input chunk at stage 0.
        """
        req_id = session.request_id
        self._new_prompt_len_snapshot[req_id] = len(update.prompt_token_ids)
        if self.vllm_config.model_config.stage_id != 0:
            self._replace_session_with_streaming_update(session, update)

        else:
            super()._update_request_as_session(session, update)

    def _free_request(self, request: Request, delay_free_blocks: bool = False) -> dict[str, Any] | None:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""
        assert request.is_finished()

        self._omits_kv_transfer_cache.pop(request.request_id, None)
        self._baseline_forget_request_ids(request.request_id)
        self._omni_recovery_trace_by_request.pop(request.request_id, None)

        # 1. Standard cleanup parts from base _free_request
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        self._new_prompt_len_snapshot.pop(request_id, None)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # 2. Omni Specific: Check if we need to transfer KV
        if self._should_transfer_kv_for_request(request_id):
            already_triggered = request_id in self.transfer_triggered_requests
            is_active = request_id in self.active_kv_transfers

            if already_triggered:
                if is_active:
                    # It triggered but hasn't finished yet. We MUST wait.
                    logger.debug(f"[Omni] Request {request_id} finished but transfer is still ACTIVE. Waiting.")
                    self.waiting_for_transfer_free.add(request_id)
                    kv_xfer_params = None
                    return kv_xfer_params
                elif request_id in self.waiting_for_transfer_free:
                    # Blocks held until KV extraction completes in a future step.
                    return None
                else:
                    logger.debug(
                        f"[Omni] Request {request_id} finished and transfer no longer ACTIVE (extracted/acked). "
                        "Freeing immediately."
                    )
            else:
                self.waiting_for_transfer_free.add(request_id)
                confirmed_computed = self._get_confirmed_num_computed_tokens(request)
                self._mark_request_for_kv_transfer(request_id, confirmed_computed)
                # Return KV transfer metadata so it propagates to RequestOutput
                if request_id in self.requests_needing_kv_transfer:
                    transfer_data = self.requests_needing_kv_transfer[request_id]
                    kv_xfer_params = {
                        "past_key_values": transfer_data["block_ids"],
                        "kv_metadata": {"seq_len": transfer_data["seq_len"], "block_ids": transfer_data["block_ids"]},
                    }
                    # Also update request.additional_information for good measure
                    add_info = getattr(request, "additional_information", None)
                    # If additional_information is an AdditionalInformationPayload-like object,
                    # unpack it into a plain dict.
                    if (
                        add_info is not None
                        and hasattr(add_info, "entries")
                        and isinstance(getattr(add_info, "entries"), dict)
                    ):
                        request.additional_information = deserialize_additional_information(add_info)
                        add_info = request.additional_information
                    if add_info is None:
                        request.additional_information = {}
                        add_info = request.additional_information
                    if isinstance(add_info, dict):
                        add_info.update(kv_xfer_params)

                return kv_xfer_params

        # 3. Standard Freeing
        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        # Helper to match base class structure if not directly available
        # VLLMScheduler has _free_blocks
        super()._free_blocks(request)

    def _mark_request_for_kv_transfer(self, req_id: str, seq_len: int) -> None:
        """Mark a request as needing KV cache transfer when it finishes."""
        # Avoid duplicate marking (if already pending in queue)
        if req_id in self.requests_needing_kv_transfer:
            return

        if self._should_transfer_kv_for_request(req_id):
            # [Omni] Get block IDs from KVCacheManager
            try:
                block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                if block_ids_tuple and len(block_ids_tuple) > 0:
                    block_ids = block_ids_tuple[0]

                    # [Omni] Fix: Truncate blocks to match seq_len snapshot
                    # We need to know block_size. Usually in self.cache_config.block_size
                    # Note: vllm_config might not be directly available, check scheduler_config or cache_config
                    if hasattr(self, "cache_config") and hasattr(self.cache_config, "block_size"):
                        block_size = self.cache_config.block_size
                    elif hasattr(self, "scheduler_config") and hasattr(
                        self.scheduler_config, "block_size"
                    ):  # Some versions
                        block_size = self.scheduler_config.block_size
                    else:
                        raise ValueError("Block size not found in cache_config or scheduler_config")

                    # ceil(seq_len / block_size)
                    num_blocks = (seq_len + block_size - 1) // block_size
                    if len(block_ids) > num_blocks:
                        logger.debug(
                            f"[Omni] Truncating blocks for {req_id} from {len(block_ids)} "
                            f"to {num_blocks} (seq_len={seq_len})"
                        )
                        block_ids = block_ids[:num_blocks]

                else:
                    block_ids = []
            except Exception as e:
                init_logger(__name__).warning(f"Failed to get block IDs for {req_id}: {e}")
                block_ids = []

            self.requests_needing_kv_transfer[req_id] = {"seq_len": seq_len, "block_ids": block_ids}
            logger.debug(f"Marked request {req_id} for KV cache transfer (len={seq_len}, blocks={len(block_ids)})")

    def _should_transfer_kv_for_request(self, req_id: str) -> bool:
        """Determine if a request should trigger KV cache transfer."""
        need_send = False
        # Try to read from vLLM Config (where YAML config is typically loaded)
        # Check for omni_kv_config attribute
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            # omni_kv_config could be an object or a dict
            if isinstance(omni_kv_config, dict):
                need_send = omni_kv_config.get("need_send_cache", False)
            else:
                need_send = getattr(omni_kv_config, "need_send_cache", False)
        if not need_send:
            return False
        request = self.requests.get(req_id)
        if request is not None and self._request_omits_kv_transfer_to_next_stage(request):
            return False
        return True

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        # [Omni] Also check for pending KV transfers
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        # [Omni] Also check for pending KV transfers to ensure the engine loop continues
        # MUST verify waiting_for_transfer_free and active_kv_transfers
        # Otherwise engine loop might exit before transfer Ack is received.
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()

        # Mark these requests as ACTIVE (sent to runner)
        self.active_kv_transfers.update(requests.keys())

        self.requests_needing_kv_transfer.clear()
        return requests


class OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler):
    """Asynchronous AutoRegressive scheduler."""
