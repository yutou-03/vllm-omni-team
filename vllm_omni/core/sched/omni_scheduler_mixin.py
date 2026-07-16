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
    emit_iteration_event,
    emit_stage_event,
)
from vllm_omni.scheduling.metadata import (
    extract_scheduling_metadata,
    merge_scheduling_metadata_into_additional_information,
)

_STATS_INTERVAL_S = 1.0


def preserve_streaming_scheduling_metadata(
    existing_additional_information: Any,
    incoming_additional_information: Any,
) -> Any:
    """Keep server-owned scheduling fields across a streaming replacement."""

    existing = deserialize_additional_information(
        existing_additional_information
    )
    scheduling_metadata = extract_scheduling_metadata(existing)
    if not scheduling_metadata:
        return incoming_additional_information or None
    incoming = deserialize_additional_information(
        incoming_additional_information
    )
    merged = merge_scheduling_metadata_into_additional_information(
        incoming,
        scheduling_metadata,
    )
    return serialize_additional_information(merged)


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    def _omni_stage_id_for_trace(self) -> int | str:
        return getattr(self.vllm_config.model_config, "stage_id", "?")

    def _omni_next_iteration_id(self) -> int:
        iteration_id = int(getattr(self, "_omni_stage_queue_iteration_id", 0)) + 1
        self._omni_stage_queue_iteration_id = iteration_id
        return iteration_id

    def add_request(self, request: Request, *args, **kwargs) -> None:
        stage_id = self._omni_stage_id_for_trace()
        request_id = getattr(request, "request_id", None)
        emit_stage_event(
            "server_receive",
            stage_id=stage_id,
            request_id=request_id,
            queue_len=len(self.waiting),
            active_reqs=len(self.running),
        )
        emit_stage_event(
            "stage_enqueue",
            stage_id=stage_id,
            request_id=request_id,
            queue_len=len(self.waiting),
            active_reqs=len(self.running),
        )
        if str(stage_id) == "2" and request_id is not None:
            req_id = str(request_id)
            with nvtx_range(f"TTFP:s2_enqueue:req={req_id[-8:]}", color="cyan"):
                result = super().add_request(request, *args, **kwargs)
        else:
            result = super().add_request(request, *args, **kwargs)
        return result

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
        )
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
