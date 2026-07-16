from __future__ import annotations

import time
from typing import Any

from vllm.v1.engine import EngineCoreEventType
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.profiling.nvtx import nvtx_range
from vllm_omni.profiling.stage_queue_trace import (
    ConformanceIneligibleReason,
    conformance_trace_enabled,
    emit_iteration_event,
    emit_stage_event,
)
from vllm_omni.scheduling.metadata import (
    extract_scheduling_metadata,
    preserve_scheduling_metadata,
)
from vllm_omni.scheduling.policy import (
    BaselineSchedulingPolicy,
    get_baseline_scheduling_policy,
    policy_applies_to_stage,
    policy_key,
)
from vllm_omni.scheduling.request_queue import maybe_create_policy_ordered_queue

_STATS_INTERVAL_S = 1.0


def preserve_streaming_scheduling_metadata(
    existing_additional_information: Any,
    incoming_additional_information: Any,
) -> Any:
    """Keep server-owned scheduling fields across a streaming replacement."""

    existing = deserialize_additional_information(
        existing_additional_information
    )
    if not extract_scheduling_metadata(existing):
        return incoming_additional_information or None
    incoming = deserialize_additional_information(
        incoming_additional_information
    )
    merged = preserve_scheduling_metadata(existing, incoming)
    return serialize_additional_information(merged)


def extract_request_scheduling_trace_fields(request: Any) -> dict[str, Any]:
    """Decode the immutable scheduling fields recorded at stage admission."""

    decoded = deserialize_additional_information(
        getattr(request, "additional_information", None)
    )
    return extract_scheduling_metadata(decoded) or {}


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    def _initialize_baseline_scheduling(self) -> None:
        """Configure optional baseline ordering after the native scheduler init."""

        self._baseline_policy = get_baseline_scheduling_policy()
        self._baseline_stage_id = int(self._omni_stage_id_for_trace())
        self._baseline_data_ready_time: dict[str, float] = {}
        self._baseline_conformance_snapshot: dict[str, Any] | None = None

        if not self._baseline_policy_applies():
            return
        if self._baseline_policy is BaselineSchedulingPolicy.SRPF_LOCAL_NP:
            raise RuntimeError(
                "srpf_local_np is disabled until a frozen stage predictor "
                "passes the calibration gate"
            )

        self.waiting = maybe_create_policy_ordered_queue(
            self.waiting,
            policy=self._baseline_policy,
            stage_id=self._baseline_stage_id,
            data_ready_time=self._baseline_request_data_ready_time,
        )
        self.skipped_waiting = maybe_create_policy_ordered_queue(
            self.skipped_waiting,
            policy=self._baseline_policy,
            stage_id=self._baseline_stage_id,
            data_ready_time=self._baseline_request_data_ready_time,
        )

    def _baseline_policy_applies(self) -> bool:
        policy = getattr(
            self,
            "_baseline_policy",
            BaselineSchedulingPolicy.NATIVE_FCFS,
        )
        stage_id = int(
            getattr(
                self,
                "_baseline_stage_id",
                self._omni_stage_id_for_trace(),
            )
        )
        return policy_applies_to_stage(policy, stage_id)

    def _baseline_request_data_ready_time(self, request: Request) -> float:
        return self._baseline_data_ready_time.get(request.request_id, float("inf"))

    def _baseline_request_key(self, request: Request):
        return policy_key(
            request,
            policy=self._baseline_policy,
            stage_id=self._baseline_stage_id,
            data_ready_time=self._baseline_request_data_ready_time(request),
            now=time.monotonic(),
        )

    def _baseline_request_is_ready_on_add(self) -> bool:
        if self._baseline_stage_id == 0:
            return True
        return getattr(self, "chunk_transfer_adapter", None) is None

    def _baseline_validate_and_track_admission(self, request: Request) -> None:
        if not self._baseline_policy_applies():
            return
        if self._baseline_request_is_ready_on_add():
            self._baseline_data_ready_time.setdefault(
                request.request_id,
                time.monotonic(),
            )
        self._baseline_request_key(request)

    def _baseline_prepare_schedule(
        self,
        *,
        token_budget_before: int | None = None,
    ) -> None:
        """Refresh data-ready times and order runnable requests in place."""

        if self._baseline_policy_applies():
            adapter = getattr(self, "chunk_transfer_adapter", None)
            if adapter is not None:
                ready_time = time.monotonic()
                for request_id in adapter.requests_with_ready_chunks:
                    self._baseline_data_ready_time.setdefault(request_id, ready_time)
            self.running.sort(key=self._baseline_request_key)
        self._baseline_capture_conformance_snapshot(
            token_budget_before=token_budget_before,
        )

    def _baseline_capture_conformance_snapshot(
        self,
        *,
        token_budget_before: int | None,
    ) -> None:
        """Freeze scheduler inputs needed to replay one policy decision."""

        if not conformance_trace_enabled():
            self._baseline_conformance_snapshot = None
            return

        running = list(self.running)
        waiting = list(self.waiting)
        skipped_waiting = list(getattr(self, "skipped_waiting", ()))
        max_running = int(getattr(self, "max_num_running_reqs", len(running)))
        sequence_slots = max(max_running - len(running), 0)
        if token_budget_before is None:
            token_budget_before = int(
                getattr(self, "max_num_scheduled_tokens", 0)
            )

        ineligible_reasons: dict[str, str] = {}
        request_locations: dict[str, str] = {}
        policy_domains: dict[str, str] = {}
        ordered_requests: list[Request] = []
        seen_request_ids: set[str] = set()

        def add_requests(requests: list[Request], location: str) -> None:
            for request in requests:
                request_id = str(request.request_id)
                if request_id in seen_request_ids:
                    continue
                seen_request_ids.add(request_id)
                request_locations[request_id] = location
                policy_domains[request_id] = (
                    "running" if location == "running" else "waiting"
                )
                status_name = getattr(
                    getattr(request, "status", None),
                    "name",
                    str(getattr(request, "status", "")),
                ).lower()
                if "finished" in status_name:
                    ineligible_reasons[request_id] = (
                        ConformanceIneligibleReason.FINISHED.value
                    )
                    continue
                if "waiting_for_input" in status_name:
                    ineligible_reasons[request_id] = (
                        ConformanceIneligibleReason.WAITING_FOR_INPUT.value
                    )
                    continue
                if "waiting_for_chunk" in status_name:
                    ineligible_reasons[request_id] = (
                        ConformanceIneligibleReason.WAITING_FOR_CHUNK.value
                    )
                    continue
                if location != "running" and sequence_slots == 0:
                    ineligible_reasons[request_id] = (
                        ConformanceIneligibleReason.NONPREEMPTIVE_RUNNING_CAPACITY.value
                    )
                    continue
                if token_budget_before <= 0:
                    ineligible_reasons[request_id] = (
                        ConformanceIneligibleReason.TOKEN_BUDGET_EXHAUSTED.value
                    )
                    continue
                ordered_requests.append(request)

        add_requests(running, "running")
        add_requests(waiting, "waiting")
        add_requests(skipped_waiting, "skipped_waiting")

        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is not None:
            for attribute in (
                "waiting_for_chunk_running_requests",
                "waiting_for_chunk_waiting_requests",
            ):
                for request in list(getattr(adapter, attribute, ())):
                    request_id = str(request.request_id)
                    ineligible_reasons[request_id] = (
                        ConformanceIneligibleReason.WAITING_FOR_CHUNK.value
                    )
                    request_locations.setdefault(request_id, attribute)

        runnable_req_ids = [str(request.request_id) for request in ordered_requests]
        policy_keys: dict[str, list[float | int | str]] = {}
        if self._baseline_policy_applies():
            for request in ordered_requests:
                policy_keys[str(request.request_id)] = list(
                    self._baseline_request_key(request)
                )
            policy_key_kind = "baseline_policy_key"
        else:
            domain_offsets = {"running": 0, "waiting": 0}
            for request in ordered_requests:
                request_id = str(request.request_id)
                domain = policy_domains[request_id]
                policy_keys[request_id] = [
                    domain,
                    domain_offsets[domain],
                ]
                domain_offsets[domain] += 1
            policy_key_kind = "native_queue_order"

        self._baseline_conformance_snapshot = {
            "policy": self._baseline_policy.value,
            "policy_applies_to_stage": self._baseline_policy_applies(),
            "policy_key_kind": policy_key_kind,
            "runnable_req_ids": runnable_req_ids,
            "policy_keys": policy_keys,
            "policy_domains": policy_domains,
            "request_locations": request_locations,
            "ineligible_reasons": ineligible_reasons,
            "token_budget_before": token_budget_before,
            "sequence_slots_before": sequence_slots,
        }

    def _baseline_forget_request_ids(self, request_ids: Any) -> None:
        ready_times = getattr(self, "_baseline_data_ready_time", None)
        if ready_times is None:
            return
        if request_ids is None:
            ready_times.clear()
            return
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        for request_id in request_ids:
            ready_times.pop(request_id, None)

    def _omni_stage_id_for_trace(self) -> int | str:
        return getattr(self.vllm_config.model_config, "stage_id", "?")

    def _omni_next_iteration_id(self) -> int:
        iteration_id = int(getattr(self, "_omni_stage_queue_iteration_id", 0)) + 1
        self._omni_stage_queue_iteration_id = iteration_id
        return iteration_id

    def add_request(self, request: Request, *args, **kwargs) -> None:
        stage_id = self._omni_stage_id_for_trace()
        request_id = getattr(request, "request_id", None)
        scheduling_fields = extract_request_scheduling_trace_fields(request)
        self._baseline_validate_and_track_admission(request)
        emit_stage_event(
            "server_receive",
            stage_id=stage_id,
            request_id=request_id,
            queue_len=len(self.waiting),
            active_reqs=len(self.running),
            **scheduling_fields,
        )
        emit_stage_event(
            "stage_enqueue",
            stage_id=stage_id,
            request_id=request_id,
            queue_len=len(self.waiting),
            active_reqs=len(self.running),
            **scheduling_fields,
        )
        if str(stage_id) == "2" and request_id is not None:
            req_id = str(request_id)
            with nvtx_range(f"TTFP:s2_enqueue:req={req_id[-8:]}", color="cyan"):
                result = super().add_request(request, *args, **kwargs)
        else:
            result = super().add_request(request, *args, **kwargs)
        return result

    def _select_waiting_queue_for_scheduling(self):
        if not self._baseline_policy_applies():
            return super()._select_waiting_queue_for_scheduling()
        if self.waiting and self.skipped_waiting:
            waiting_key = self._baseline_request_key(
                self.waiting.peek_request()
            )
            skipped_key = self._baseline_request_key(
                self.skipped_waiting.peek_request()
            )
            return self.waiting if waiting_key <= skipped_key else self.skipped_waiting
        return self.waiting or self.skipped_waiting or None

    def _free_request(self, request: Request, *args, **kwargs):
        try:
            return super()._free_request(request, *args, **kwargs)
        finally:
            self._baseline_forget_request_ids(request.request_id)

    def _trace_scheduler_output(
        self,
        scheduler_output,
        *,
        iteration_id: int,
        timestamp_start: float,
        timestamp_end: float,
        num_running_before: int,
        num_waiting_before: int,
    ) -> None:
        stage_id = self._omni_stage_id_for_trace()
        schedule_duration_ms = (timestamp_end - timestamp_start) * 1000.0
        num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        scheduled_req_ids = [str(rid) for rid in num_scheduled_tokens.keys()]
        total_num_scheduled_tokens = int(
            getattr(
                scheduler_output,
                "total_num_scheduled_tokens",
                sum(int(v) for v in num_scheduled_tokens.values()),
            )
            or 0
        )
        conformance = self._baseline_conformance_snapshot or {}
        runnable_req_ids = list(conformance.get("runnable_req_ids", ()))
        runnable_set = set(runnable_req_ids)
        selected_set = set(scheduled_req_ids) & runnable_set
        policy_domains = conformance.get("policy_domains", {})
        policy_activation = False
        for domain in set(policy_domains.values()):
            domain_ids = {
                request_id
                for request_id in runnable_set
                if policy_domains.get(request_id) == domain
            }
            if (
                len(domain_ids) >= 2
                and selected_set & domain_ids
                and domain_ids - selected_set
            ):
                policy_activation = True
                break
        token_budget_before = conformance.get("token_budget_before")
        token_budget_after = (
            max(int(token_budget_before) - total_num_scheduled_tokens, 0)
            if token_budget_before is not None
            else None
        )
        max_running = int(getattr(self, "max_num_running_reqs", len(self.running)))
        conformance_fields = dict(conformance)
        if conformance:
            conformance_fields.update(
                selected_req_ids=scheduled_req_ids,
                num_scheduled_tokens={
                    str(request_id): int(num_tokens)
                    for request_id, num_tokens in num_scheduled_tokens.items()
                },
                token_budget_after=token_budget_after,
                sequence_slots_after=max(max_running - len(self.running), 0),
                policy_activation=policy_activation,
            )
        emit_iteration_event(
            stage_id=stage_id,
            iteration_id=iteration_id,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_running_reqs_before=num_running_before,
            num_waiting_reqs_before=num_waiting_before,
            batch_num_tokens=total_num_scheduled_tokens,
            batch_num_seqs=len(num_scheduled_tokens),
            scheduled_req_ids=scheduled_req_ids,
            preempted_req_ids=list(getattr(scheduler_output, "preempted_req_ids", set()) or []),
            kv_cache_usage=getattr(self.kv_cache_manager, "usage", None),
            **conformance_fields,
        )
        self._baseline_conformance_snapshot = None
        emit_stage_event(
            "stage_schedule_done",
            stage_id=stage_id,
            iteration_id=iteration_id,
            duration_ms=schedule_duration_ms,
            queue_len=num_waiting_before,
            active_reqs=num_running_before,
            queue_len_after=len(self.waiting),
            active_reqs_after=len(self.running),
            batch_num_tokens=total_num_scheduled_tokens,
            batch_num_seqs=len(num_scheduled_tokens),
            scheduled_req_ids=scheduled_req_ids,
            preempted_req_ids=list(getattr(scheduler_output, "preempted_req_ids", set()) or []),
            kv_cache_usage=getattr(self.kv_cache_manager, "usage", None),
        )
        for rid, num_tokens in num_scheduled_tokens.items():
            rid_str = str(rid)
            emit_stage_event(
                "stage_schedule",
                stage_id=stage_id,
                request_id=rid_str,
                iteration_id=iteration_id,
                queue_len=num_waiting_before,
                active_reqs=num_running_before,
                queue_len_after=len(self.waiting),
                active_reqs_after=len(self.running),
                batch_num_tokens=total_num_scheduled_tokens,
                batch_num_seqs=len(num_scheduled_tokens),
                num_scheduled_tokens=int(num_tokens),
                schedule_duration_ms=schedule_duration_ms,
            )
            if str(stage_id) == "2":
                first_schedule_seen = getattr(self, "_omni_nvtx_s2_first_schedule_seen", set())
                if rid_str not in first_schedule_seen:
                    first_schedule_seen.add(rid_str)
                    with nvtx_range(f"TTFP:s2_first_schedule:req={rid_str[-8:]}", color="orange"):
                        self._omni_nvtx_s2_first_schedule_seen = first_schedule_seen

    def _replace_session_with_streaming_update(
        self,
        session: Request,
        update: StreamingUpdate,
    ) -> None:
        """For streaming input: Replace an existing streaming session payload with the latest update."""
        session._output_token_ids.clear()
        session._all_token_ids.clear()
        new_prompt = update.prompt_token_ids or ()
        session._all_token_ids.extend(new_prompt)
        session.num_computed_tokens = 0
        session.prompt_token_ids = update.prompt_token_ids or ()
        session.additional_information = preserve_streaming_scheduling_metadata(
            session.additional_information,
            update.additional_information,
        )
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        # now = time.monotonic()
        # if now - getattr(self, "_last_stats_time", 0.0) < _STATS_INTERVAL_S:
        #     return None
        # self._last_stats_time = now
        kv_cache_usage = kwargs.pop("kv_cache_usage_before_update", self.kv_cache_manager.usage)
        num_running = kwargs.pop("num_running_before_update", len(self.running))
        num_waiting = kwargs.pop("num_waiting_before_update", len(self.waiting))
        return SchedulerStats(
            kv_cache_usage=kv_cache_usage,
            num_running_reqs=num_running,
            num_waiting_reqs=num_waiting,
        ) 
