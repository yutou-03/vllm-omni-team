"""
Orchestrator for vLLM-Omni multi-stage runtime.

Runs inside a background thread with its own asyncio event loop.
Owns logical request progression across stage pools and handles
stage-to-stage transfer logic.
"""

from __future__ import annotations

import asyncio
import copy
import time as _time
from dataclasses import dataclass, field
from typing import Any, Sequence

import janus
import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.metrics.stats import IterationStats

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.metrics.prometheus import OmniRequestCounter
from vllm_omni.metrics.stat_logger import OmniPrometheusStatLogger
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.profiling.nvtx import (
    nvtx_mark,
    nvtx_range,
)
from vllm_omni.profiling.stage_queue_trace import emit_stage_event, stage_trace_span
from vllm_omni.recovery import (
    ExecutionLedger,
    MessageDropReason,
    OutputCommitLedger,
    OutputDecision,
    PreemptionRecoveryMode,
)
from vllm_omni.scheduling.config import (
    SloSchedulingMode,
    extract_slo_scheduling_config,
)
from vllm_omni.scheduling.contracts import RequestSLOContract
from vllm_omni.scheduling.metadata import (
    extract_scheduling_metadata,
    merge_scheduling_metadata_into_additional_information,
    merge_scheduling_metadata_into_prompt,
)

logger = init_logger(__name__)

_SLO_PRESSURE_FEEDBACK_INTERVAL_S = 0.1
_SLO_PRESSURE_FEEDBACK_TIMEOUT_S = 0.25


def extract_scheduling_metadata_from_request_input(
    request_input: Any,
) -> dict[str, Any] | None:
    """Extract canonical scheduling fields from a prompt or core request."""

    if isinstance(request_input, list):
        for item in request_input:
            if scheduling := extract_scheduling_metadata_from_request_input(item):
                return scheduling
        return None
    if isinstance(request_input, dict):
        return extract_scheduling_metadata(
            request_input.get("additional_information")
        )
    wire_payload = getattr(request_input, "additional_information", None)
    return extract_scheduling_metadata(
        deserialize_additional_information(wire_payload)
    )


def merge_scheduling_metadata_into_request_input(
    request_input: Any,
    scheduling_metadata: dict[str, Any] | None,
) -> Any:
    """Preserve scheduling fields when an inter-stage input is rebuilt."""

    if not scheduling_metadata:
        return request_input
    if isinstance(request_input, list):
        return [
            merge_scheduling_metadata_into_request_input(
                item,
                scheduling_metadata,
            )
            for item in request_input
        ]
    if isinstance(request_input, dict):
        return merge_scheduling_metadata_into_prompt(
            request_input,
            scheduling_metadata,
        )
    if not hasattr(request_input, "additional_information"):
        return request_input
    decoded = deserialize_additional_information(
        request_input.additional_information
    )
    merged = merge_scheduling_metadata_into_additional_information(
        decoded,
        scheduling_metadata,
    )
    return OmniEngineCoreRequest.from_request(
        request_input,
        additional_information=serialize_additional_information(merged),
    )


def set_execution_global_request_id(
    request_input: Any,
    execution_id: str,
) -> Any:
    """Give a restarted execution a fresh model/KV-transfer identity."""
    if isinstance(request_input, dict):
        additional_information = request_input.get("additional_information")
        if additional_information is None:
            additional_information = {}
        elif not isinstance(additional_information, dict):
            additional_information = deserialize_additional_information(
                additional_information
            )
        additional_information["global_request_id"] = [execution_id]
        request_input["additional_information"] = additional_information
        return request_input

    if not hasattr(request_input, "additional_information"):
        return request_input
    additional_information = deserialize_additional_information(
        request_input.additional_information
    )
    additional_information["global_request_id"] = [execution_id]
    request_input.additional_information = serialize_additional_information(
        additional_information
    )
    return request_input


def merge_pipeline_attempt_metadata_into_request_input(
    request_input: Any,
    *,
    logical_request_id: str,
    execution_id: str,
    pipeline_attempt_id: int,
    stage_attempt_id: int,
    recovery_target_stage_id: int,
    recovery_target_num_computed_tokens: int,
    recovery_preemption_cause: str,
    controlled_rule_id: str | None,
) -> Any:
    """Attach immutable recovery trace fields to one physical execution."""

    recovery_meta: dict[str, Any] = {
        "omni_logical_request_id": logical_request_id,
        "omni_execution_id": execution_id,
        "omni_pipeline_attempt_id": int(pipeline_attempt_id),
        "omni_stage_attempt_id": int(stage_attempt_id),
        "omni_recovery_target_stage_id": int(recovery_target_stage_id),
        "omni_recovery_target_num_computed_tokens": int(
            recovery_target_num_computed_tokens
        ),
        "omni_recovery_preemption_cause": recovery_preemption_cause,
    }
    if controlled_rule_id is not None:
        recovery_meta["omni_controlled_rule_id"] = controlled_rule_id

    if isinstance(request_input, list):
        return [
            merge_pipeline_attempt_metadata_into_request_input(
                item,
                logical_request_id=logical_request_id,
                execution_id=execution_id,
                pipeline_attempt_id=pipeline_attempt_id,
                stage_attempt_id=stage_attempt_id,
                recovery_target_stage_id=recovery_target_stage_id,
                recovery_target_num_computed_tokens=(
                    recovery_target_num_computed_tokens
                ),
                recovery_preemption_cause=recovery_preemption_cause,
                controlled_rule_id=controlled_rule_id,
            )
            for item in request_input
        ]

    if isinstance(request_input, dict):
        merged_input = dict(request_input)
        additional_information = dict(
            deserialize_additional_information(
                merged_input.get("additional_information")
            )
        )
        meta = dict(additional_information.get("meta") or {})
        meta.update(recovery_meta)
        additional_information["meta"] = meta
        merged_input["additional_information"] = additional_information
        return merged_input

    if not hasattr(request_input, "additional_information"):
        return request_input
    additional_information = dict(
        deserialize_additional_information(
            request_input.additional_information
        )
    )
    meta = dict(additional_information.get("meta") or {})
    meta.update(recovery_meta)
    additional_information["meta"] = meta
    request_input.additional_information = serialize_additional_information(
        additional_information
    )
    return request_input


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
    resumable: bool = False,
    mm_features: list | None = None,
    connector_req_id: str | None = None,
    output_connector_req_id: str | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt."""
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    prompt_embeds: torch.Tensor | None = prompt.get("prompt_embeds")
    additional_info_payload = serialize_additional_information(
        prompt.get("additional_information"),
        log_prefix=f"build_engine_core_request_from_tokens req={request_id}",
    )

    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        arrival_time=arrival_time,
        lora_request=getattr(params, "lora_request", None),
        cache_salt=None,
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds,
        resumable=resumable,
        additional_information=additional_info_payload,
        connector_req_id=connector_req_id,
        output_connector_req_id=output_connector_req_id,
    )


@dataclass
class OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    logical_request_id: str | None = None
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1

    # Metrics: timestamp when request was submitted to each stage.
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    mm_processor_kwargs: dict | None = None
    mm_features: list | None = None
    pd_prefill_multimodal_output: dict[str, Any] | None = None
    scheduling_metadata: dict[str, Any] | None = None
    slo_contract: RequestSLOContract | None = None

    streaming: StreamingInputState = field(default_factory=lambda: StreamingInputState())

    # Per-request pipeline timing accumulator (milliseconds)
    pipeline_timings: dict[str, float] = field(default_factory=dict)

    # Recovery bookkeeping.  A resource preemption invalidates the current
    # physical pipeline and creates a new execution id from the Stage-0 input.
    final_outputs_emitted: dict[int, int] = field(default_factory=dict)
    replay_outputs_to_drop: dict[int, int] = field(default_factory=dict)
    discard_stage_outputs_before: dict[int, float] = field(default_factory=dict)
    stage0_request_template: Any = None
    current_stage0_request: Any = None
    stage0_prompt_text: str | None = None
    stage_request_ids: dict[int, str] = field(default_factory=dict)
    pending_restart_stages: set[int] = field(default_factory=set)
    execution_attempt: int = 0
    current_execution_id: str | None = None
    recovery_target_stage_id: int | None = None
    recovery_target_progress: int | None = None
    recovery_preemption_cause: str | None = None
    recovery_controlled_rule_id: str | None = None
    execution_ledger: ExecutionLedger | None = None
    output_commit_ledger: OutputCommitLedger | None = None
    output_attempt_sequences: dict[int, int] = field(default_factory=dict)
    # Backend generation order need not equal frontend queue-consumption order.
    # Only processed terminal envelopes from the current execution count here.
    processed_terminal_stages: set[int] = field(default_factory=set)
    completion_deferred: bool = False

    def outputs_drained(self) -> bool:
        return (
            self.final_stage_id in self.processed_terminal_stages
            and not self.pending_restart_stages
            and set(self.stage_submit_ts).issubset(self.processed_terminal_stages)
        )

    def __post_init__(self) -> None:
        """Create the explicit execution ledger for a valid pipeline request."""

        if self.logical_request_id is None:
            self.logical_request_id = self.request_id
        if self.scheduling_metadata is not None and self.scheduling_metadata.get(
            "sched_schema_version"
        ) == 2:
            parsed_contract = RequestSLOContract.from_server_metadata(
                self.scheduling_metadata
            )
            if parsed_contract.logical_request_id != self.logical_request_id:
                raise ValueError(
                    "logical request id mismatch between orchestrator state "
                    "and schema-v2 SLO contract"
                )
            if self.slo_contract is None:
                self.slo_contract = parsed_contract
            elif self.slo_contract != parsed_contract:
                raise ValueError(
                    "schema-v2 scheduling metadata and SLO contract differ"
                )
        elif (
            self.slo_contract is not None
            and self.slo_contract.logical_request_id != self.logical_request_id
        ):
            raise ValueError(
                "logical request id mismatch between orchestrator state "
                "and SLO contract"
            )

        if self.final_stage_id < 0:
            return
        if self.execution_ledger is None:
            initial_physical_id = self.current_execution_id or self.request_id
            physical_request_ids = {
                stage_id: self.stage_request_ids.get(stage_id, initial_physical_id)
                for stage_id in range(self.final_stage_id + 1)
            }
            self.execution_ledger = ExecutionLedger.initialize(
                logical_request_id=self.logical_request_id,
                final_stage_id=self.final_stage_id,
                physical_request_ids=physical_request_ids,
                started_monotonic_s=_time.monotonic(),
            )
        if self.output_commit_ledger is None:
            self.output_commit_ledger = OutputCommitLedger.create(
                self._output_stream_id(stage_id)
                for stage_id in range(self.final_stage_id + 1)
            )
        for stage_id in range(self.final_stage_id + 1):
            self.output_attempt_sequences.setdefault(stage_id, 0)

    @staticmethod
    def _output_stream_id(stage_id: int) -> str:
        return f"stage:{stage_id}"

    def begin_output_replay(self, stage_ids: list[int]) -> None:
        """Snapshot frontend commits and reset per-attempt output sequences."""

        if self.output_commit_ledger is None:
            return
        replay_stream_ids = []
        for stage_id in dict.fromkeys(stage_ids):
            stream_id = self._output_stream_id(stage_id)
            stream_state = self.output_commit_ledger.states[stream_id]
            emitted = stream_state.committed_sequence + 1
            self.replay_outputs_to_drop[stage_id] = emitted
            self.output_attempt_sequences[stage_id] = 0
            if not stream_state.terminal_committed:
                replay_stream_ids.append(stream_id)
        if replay_stream_ids:
            self.output_commit_ledger.begin_replay(replay_stream_ids)

    def classify_and_commit_output(
        self,
        stage_id: int,
        *,
        finished: bool = False,
    ) -> OutputDecision:
        """Classify the next output from the current stage attempt."""

        if self.output_commit_ledger is None:
            return OutputDecision.COMMIT
        sequence = self.output_attempt_sequences.get(stage_id, 0)
        self.output_attempt_sequences[stage_id] = sequence + 1
        decision = self.output_commit_ledger.classify_and_commit(
            self._output_stream_id(stage_id),
            sequence,
            finished=finished,
        )
        if decision in {
            OutputDecision.DROP_REPLAY_DUPLICATE,
            OutputDecision.COMMIT_TERMINAL_ONLY,
        }:
            self.replay_outputs_to_drop[stage_id] = max(
                0,
                self.replay_outputs_to_drop.get(stage_id, 0) - 1,
            )
        elif decision is OutputDecision.COMMIT:
            self.final_outputs_emitted[stage_id] = (
                self.output_commit_ledger.states[
                    self._output_stream_id(stage_id)
                ].committed_sequence
                + 1
            )
        return decision

    def trace_identity_fields(self, stage_id: int) -> dict[str, Any]:
        """Return the current canonical trace identity for one stage."""

        logical_request_id = self.logical_request_id or self.request_id
        pipeline_attempt_id = self.execution_attempt
        stage_attempt_id = 0
        if self.execution_ledger is not None:
            pipeline_attempt_id = self.execution_ledger.pipeline_attempt_id
            stage_ref = self.execution_ledger.current_stages.get(stage_id)
            if stage_ref is not None:
                stage_attempt_id = stage_ref.stage_attempt_id
        return {
            "logical_request_id": logical_request_id,
            "execution_id": self.current_execution_id or self.request_id,
            "pipeline_attempt_id": pipeline_attempt_id,
            "stage_attempt_id": stage_attempt_id,
        }


@dataclass
class StreamingInputState:
    # Flag of streaming input request
    enabled: bool = False
    # Flag of segment of streaming input finished
    segment_finished: bool = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None
    # Model/bridge-specific runtime states (e.g., thinker->talker)
    bridge_states: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Runs inside a background thread's asyncio event loop."""

    _MAX_RAW_OUTPUTS_PER_DRAIN = 16

    def __init__(
        self,
        request_async_queue: janus.AsyncQueue[dict[str, Any]],
        output_async_queue: janus.AsyncQueue[dict[str, Any]],
        rpc_async_queue: janus.AsyncQueue[dict[str, Any]],
        stage_pools: list[StagePool],
        *,
        async_chunk: bool = False,
        pd_config: dict[str, Any] | None = None,
        running_counter: OmniRequestCounter | None = None,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue

        self.async_chunk = bool(async_chunk)
        self.num_stages = len(stage_pools)
        self.stage_pools: list[StagePool] = stage_pools

        # PD disaggregation state
        self._pd_pair: tuple[int, int] | None = None
        self._pd_bootstrap_addr: str | None = None
        self._pd_prefill_engine_id: str | None = None
        self._pd_kv_params: dict[str, Any] = {}
        if pd_config is not None:
            self._pd_pair = pd_config.get("pd_pair")
            self._pd_bootstrap_addr = pd_config.get("bootstrap_addr")
            self._pd_prefill_engine_id = pd_config.get("prefill_engine_id")
        self.request_states: dict[str, OrchestratorRequestState] = {}
        self._physical_to_logical_request_ids: dict[str, str] = {}
        self._cfg_tracker = CfgCompanionTracker()
        self._running_counter = running_counter
        self._omni_nvtx_s2_raw_output_seen: set[str] = set()
        self._omni_nvtx_s2_route_seen: set[str] = set()

        vllm_config_for_stats = next(
            (p.stage_vllm_config for p in stage_pools if p.stage_vllm_config is not None),
                None,
        )
        # Build flat engine_idx ↔ (stage_id, replica_id) maps so that the wrap
        # exposes the ~37 vllm:* families with per-(stage, replica) labels.
        # The reverse map is consulted at record() time to find the engine_idx
        # to update from the orchestrator's (stage_id, replica_id) loop.
        stage_replica_map: dict[int, tuple[str, str]] = {}
        self._stage_replica_to_engine_idx: dict[tuple[int, int], int] = {}
        flat_idx = 0
        for stage_id, pool in enumerate(stage_pools):
            for replica_id in range(pool.num_replicas):
                stage_replica_map[flat_idx] = (str(stage_id), str(replica_id))
                self._stage_replica_to_engine_idx[(stage_id, replica_id)] = flat_idx
                flat_idx += 1

        if vllm_config_for_stats is not None:
            self._stat_logger: OmniPrometheusStatLogger | None = OmniPrometheusStatLogger(
                vllm_config=vllm_config_for_stats,
                stage_replica_map=stage_replica_map,
            )
        else:
            self._stat_logger = None
        self._last_stats_ts: float = 0.0
        self._stats_interval_s: float = 1.0

        self._shutdown_event = asyncio.Event()
        self._stages_shutdown = False
        self._fatal_error: str | None = None
        self._fatal_error_stage_id: int | None = None
        self._drain_state_epoch = 0
        self._drain_output_inflight = 0
        self._slo_feedback_last_sample_by_edge: dict[
            tuple[int, int], tuple[int, float]
        ] = {}
        self._slo_feedback_error_logged_edges: set[tuple[int, int]] = set()

    def _bump_drain_epoch(self) -> None:
        """Record owner-loop state churn (also safe for minimal test doubles)."""

        self._drain_state_epoch = getattr(self, "_drain_state_epoch", 0) + 1

    def _begin_drain_output(self) -> None:
        self._drain_output_inflight = (
            getattr(self, "_drain_output_inflight", 0) + 1
        )
        self._bump_drain_epoch()

    def _end_drain_output(self) -> None:
        self._drain_output_inflight = max(
            0,
            getattr(self, "_drain_output_inflight", 0) - 1,
        )
        self._bump_drain_epoch()

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        logger.info("[Orchestrator] Starting event loop")

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(),
            name="orchestrator-stage-output-handler",
        )
        tasks = [request_task, output_task]
        if self._slo_downstream_feedback_enabled():
            tasks.append(
                asyncio.create_task(
                    self._slo_downstream_feedback_loop(),
                    name="orchestrator-slo-pressure-feedback",
                )
            )

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            self._shutdown_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass

            # If a fatal error caused the shutdown, drain any pending
            # add_request messages that were never processed and broadcast
            # fatal error responses so callers are not left hanging.
            if self._fatal_error is not None:
                await self._drain_pending_requests_on_fatal()

            self._shutdown_stages()

            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    def _slo_downstream_feedback_enabled(self) -> bool:
        """Enable the relay only when at least one dynamic stage consumes it."""

        return bool(self._slo_downstream_feedback_edges())

    def _slo_downstream_feedback_edges(self) -> tuple[tuple[int, int], ...]:
        """Select only upstream dynamic targets that require downstream data."""

        if len(self.stage_pools) < 2:
            return ()
        edges: list[tuple[int, int]] = []
        for upstream_stage_id, pool in enumerate(self.stage_pools[:-1]):
            config = getattr(pool, "stage_vllm_config", None)
            if config is None:
                continue
            try:
                slo_config = extract_slo_scheduling_config(config)
            except (TypeError, ValueError):
                continue
            if not (
                slo_config.enabled
                and slo_config.mode
                is SloSchedulingMode.STAGE_DDL_DYNAMIC_ADMISSION
            ):
                continue
            dynamic = slo_config.stage_ddl.dynamic_admission
            if dynamic is None:
                continue
            if (
                upstream_stage_id in dynamic.target_stage_ids
                and dynamic.controller.require_downstream_pressure
            ):
                edges.append((upstream_stage_id, upstream_stage_id + 1))
        return tuple(edges)

    async def _relay_slo_downstream_pressure_once(
        self,
        *,
        timeout: float = _SLO_PRESSURE_FEEDBACK_TIMEOUT_S,
        edges: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Relay adjacent single-replica pressure through owner-safe RPCs."""

        last_samples = getattr(
            self,
            "_slo_feedback_last_sample_by_edge",
            None,
        )
        if last_samples is None:
            last_samples = self._slo_feedback_last_sample_by_edge = {}
        logged_errors = getattr(
            self,
            "_slo_feedback_error_logged_edges",
            None,
        )
        if logged_errors is None:
            logged_errors = self._slo_feedback_error_logged_edges = set()

        selected_edges = (
            self._slo_downstream_feedback_edges()
            if edges is None
            else tuple(edges)
        )
        for upstream_stage_id, downstream_stage_id in reversed(selected_edges):
            edge = (upstream_stage_id, downstream_stage_id)
            upstream_pool = self.stage_pools[upstream_stage_id]
            downstream_pool = self.stage_pools[downstream_stage_id]

            # Replica-aware aggregation needs a separate, request-affine
            # contract.  Do not guess by summing unrelated replica epochs.
            if (
                upstream_pool.num_replicas != 1
                or downstream_pool.num_replicas != 1
            ):
                if edge not in logged_errors:
                    logger.warning(
                        "[Orchestrator] SLO pressure feedback disabled for "
                        "multi-replica edge %s->%s",
                        *edge,
                    )
                    logged_errors.add(edge)
                continue

            upstream_client = upstream_pool.clients[0]
            downstream_client = downstream_pool.clients[0]
            get_pressure = getattr(
                downstream_client,
                "get_slo_runtime_pressure_async",
                None,
            )
            update_pressure = getattr(
                upstream_client,
                "update_slo_downstream_pressure_async",
                None,
            )
            if not callable(get_pressure) or not callable(update_pressure):
                continue

            try:
                sample = await get_pressure(timeout=timeout)
                epoch = sample.get("control_epoch")
                captured = sample.get("captured_monotonic_s")
                if (
                    isinstance(epoch, bool)
                    or not isinstance(epoch, int)
                    or epoch < 0
                ):
                    raise ValueError(
                        "pressure sample control_epoch must be a "
                        "non-negative integer"
                    )
                if isinstance(captured, bool):
                    raise ValueError(
                        "pressure sample captured_monotonic_s must be numeric"
                    )
                try:
                    captured = float(captured)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "pressure sample captured_monotonic_s must be numeric"
                    ) from error
                previous_sample = last_samples.get(edge)
                if previous_sample is not None and (
                    epoch < previous_sample[0]
                    or captured <= previous_sample[1]
                ):
                    continue
                await update_pressure(sample, timeout=timeout)
                last_samples[edge] = (epoch, captured)
                logged_errors.discard(edge)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if edge not in logged_errors:
                    logger.warning(
                        "[Orchestrator] SLO pressure feedback failed for "
                        "edge %s->%s: %s: %s",
                        *edge,
                        type(error).__name__,
                        error,
                    )
                    logged_errors.add(edge)

    async def _slo_downstream_feedback_loop(self) -> None:
        """Periodically refresh downstream pressure without blocking routing."""

        while not self._shutdown_event.is_set():
            await self._relay_slo_downstream_pressure_once()
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=_SLO_PRESSURE_FEEDBACK_INTERVAL_S,
                )
            except TimeoutError:
                pass

    # ---- Request handling ----

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_async_queue."""
        while True:
            msg = await self.request_async_queue.get()
            msg_type = msg.get("type")

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "streaming_update":
                await self._handle_streaming_update(msg)
            elif msg_type == "add_companion_request":
                await self._handle_add_companion(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "collective_rpc":
                await self._handle_collective_rpc(msg)
            elif msg_type == "drain_status":
                try:
                    await self._handle_drain_status(msg)
                except Exception as error:
                    # Drain is an experimental read-only control RPC.  A
                    # snapshot failure must fail closed without taking down the
                    # request/output loops it is observing.
                    logger.exception("[Orchestrator] drain status failed")
                    await self.rpc_async_queue.put(
                        {
                            "type": "drain_status_result",
                            "rpc_id": msg.get("rpc_id"),
                            "schema_version": 1,
                            "healthy": False,
                            "drained": False,
                            "stable": False,
                            "replicas": [],
                            "invalid_reasons": ["snapshot_failed"],
                            "errors": [f"{type(error).__name__}: {error}"],
                        }
                    )
            elif msg_type == "shutdown":
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                self._shutdown_stages()
                break
            else:
                logger.warning("[Orchestrator] Unknown message type: %s", msg_type)

    async def _handle_add_request(self, msg: dict[str, Any]) -> None:
        """Handle an add_request message from the main thread."""
        nvtx_mark("orchestrator:handle_add_request")
        stage_id = 0
        request_id = msg["request_id"]
        prompt = msg["prompt"]
        original_prompt = msg.get("original_prompt", prompt)
        sampling_params_list = msg["sampling_params_list"]
        if not sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(sampling_params_list)} stage params.")
        final_stage_id = msg["final_stage_id"]

        logger.info(
            "[Orchestrator] _handle_add_request: stage=%s req=%s "
            "prompt_type=%s original_prompt_type=%s final_stage=%s "
            "num_sampling_params=%d",
            stage_id,
            request_id,
            type(prompt).__name__,
            type(original_prompt).__name__,
            final_stage_id,
            len(sampling_params_list),
        )

        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=original_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            mm_features=getattr(prompt, "mm_features", None),
            scheduling_metadata=(
                extract_scheduling_metadata_from_request_input(original_prompt)
                or extract_scheduling_metadata_from_request_input(prompt)
            ),
            stage0_request_template=copy.deepcopy(prompt),
            current_stage0_request=prompt,
            stage0_prompt_text=msg.get("output_prompt_text"),
            stage_request_ids={stage_id: request_id},
            current_execution_id=request_id,
        )
        self.request_states[request_id] = req_state
        self._physical_to_logical_request_ids[request_id] = request_id
        self._bump_drain_epoch()
        if self._running_counter is not None:
            self._running_counter.increment()
        req_state.streaming.enabled = bool(getattr(prompt, "resumable", False))
        req_state.stage_submit_ts[stage_id] = _time.time()
        enqueue_ts = msg.get("enqueue_ts", 0.0)
        if enqueue_ts > 0:
            req_state.pipeline_timings["queue_wait_ms"] = (_time.perf_counter() - enqueue_ts) * 1000.0
        preprocess_ms = msg.get("preprocess_ms", 0.0)
        if preprocess_ms > 0:
            req_state.pipeline_timings["preprocess_ms"] = preprocess_ms
        nvtx_mark("orchestrator:submit_initial")
        await self.stage_pools[stage_id].submit_initial(
            request_id,
            req_state,
            prompt,
            prompt_text=msg.get("output_prompt_text"),
        )

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, prompt, req_state)

    async def _handle_streaming_update(self, msg: dict[str, Any]) -> None:
        """Handle a streaming_update message for an existing request."""
        stage_id = 0
        request_id = msg["request_id"]
        request = msg["prompt"]
        nvtx_mark("orchestrator:handle_streaming_update")

        req_state = self.request_states.get(request_id)
        if req_state is None:
            logger.warning(
                "[Orchestrator] streaming_update for unknown req=%s, falling back to add_request",
                request_id,
            )
            fallback_msg = dict(msg)
            fallback_msg["type"] = "add_request"
            await self._handle_add_request(fallback_msg)
            return

        if "sampling_params_list" in msg and msg["sampling_params_list"]:
            req_state.sampling_params_list = msg["sampling_params_list"]

        request = merge_scheduling_metadata_into_request_input(
            request,
            req_state.scheduling_metadata,
        )

        req_state.streaming.enabled = True
        req_state.stage_submit_ts[stage_id] = _time.time()
        await self.stage_pools[stage_id].submit_update(
            request_id,
            req_state,
            request,
            prompt_text=msg.get("output_prompt_text"),
        )
        self._bump_drain_epoch()

    async def _handle_add_companion(self, msg: dict[str, Any]) -> None:
        """Handle an add_companion_request message: submit companion to stage 0."""
        companion_id = msg["companion_id"]
        parent_id = msg["parent_id"]
        role = msg["role"]
        companion_prompt = msg["prompt"]
        sampling_params_list = msg["sampling_params_list"]

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            logger.info(
                "[Orchestrator] Dropping CFG companion %s (role=%s): parent %s is no longer active",
                companion_id,
                role,
                parent_id,
            )
            return

        self._cfg_tracker.register_companion(parent_id, role, companion_id)

        companion_prompt = merge_scheduling_metadata_into_request_input(
            companion_prompt,
            parent_state.scheduling_metadata,
        )

        companion_state = OrchestratorRequestState(
            request_id=companion_id,
            logical_request_id=parent_state.logical_request_id,
            prompt=companion_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=0,
            scheduling_metadata=parent_state.scheduling_metadata,
        )
        self.request_states[companion_id] = companion_state
        self._bump_drain_epoch()
        companion_state.stage_submit_ts[0] = _time.time()
        companion_replica_id = await self.stage_pools[0].submit_initial(
            companion_id,
            companion_state,
            companion_prompt,
            prompt_text=msg.get("companion_prompt_text"),
            affinity_request_id=parent_id,
        )

        logger.info(
            "[Orchestrator] CFG companion submitted: %s (role=%s, parent=%s, stage-0 replica-%s)",
            companion_id,
            role,
            parent_id,
            companion_replica_id,
        )

    async def _handle_abort(self, msg: dict[str, Any]) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg["request_ids"]
        await self._cleanup_request_ids(
            self._cfg_tracker.abort_parents(request_ids),
            abort=True,
        )
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    async def _abort_request_ids(self, request_ids: list[str]) -> None:
        """Forward abort requests to all stage pools."""
        if not request_ids:
            return
        for pool in self.stage_pools:
            await pool.abort_requests(request_ids)

    def _release_request_bindings(self, request_ids: list[str]) -> None:
        """Release all stage-local route bindings for the given request ids."""
        for pool in self.stage_pools:
            pool.release_bindings(request_ids)

    async def _handle_collective_rpc(self, msg: dict[str, Any]) -> None:
        """Handle a control-plane RPC request from the main thread."""
        rpc_id = msg["rpc_id"]
        method = msg["method"]
        timeout = msg.get("timeout")
        args = tuple(msg.get("args", ()))
        kwargs = dict(msg.get("kwargs") or {})
        requested_stage_ids = msg.get("stage_ids")

        target_pools: list[StagePool] = []
        if requested_stage_ids is None:
            target_pools.extend(self.stage_pools)
        else:
            for lid in requested_stage_ids:
                if not (0 <= lid < self.num_stages):
                    logger.warning("[Orchestrator] collective_rpc: ignoring invalid stage_id %s", lid)
                    continue
                target_pools.append(self.stage_pools[lid])

        results: list[Any] = []
        stage_ids: list[int] = []
        for pool in target_pools:
            for replica_id in range(pool.num_replicas):
                stage_result = await pool.collective_rpc(
                    replica_id=replica_id,
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                )
                stage_ids.append(pool.stage_id)
                results.append(stage_result)

        await self.rpc_async_queue.put(
            {
                "type": "collective_rpc_result",
                "rpc_id": rpc_id,
                "method": method,
                "stage_ids": stage_ids,
                "results": results,
            }
        )

    def _local_drain_status(self) -> dict[str, Any]:
        """Snapshot state owned by the orchestrator event loop."""

        cfg = self._cfg_tracker.get_drain_status()
        pools = [pool.get_local_drain_status() for pool in self.stage_pools]
        janus_counters = {
            "admission_queue": self.request_async_queue.qsize(),
            "output_queue": self.output_async_queue.qsize(),
            "rpc_output_queue": self.rpc_async_queue.qsize(),
        }
        counters = {
            "request_states": len(self.request_states),
            "pd_kv_params": len(self._pd_kv_params),
            "output_handlers_inflight": getattr(
                self, "_drain_output_inflight", 0
            ),
        }
        healthy = (
            self._fatal_error is None
            and not self._stages_shutdown
            and not self._shutdown_event.is_set()
        )
        drained = (
            healthy
            and all(value == 0 for value in counters.values())
            and all(value == 0 for value in janus_counters.values())
            and bool(cfg["drained"])
            and all(bool(pool["drained"]) for pool in pools)
        )
        return {
            "epoch": getattr(self, "_drain_state_epoch", 0),
            "healthy": healthy,
            "drained": drained,
            "fatal_error": self._fatal_error,
            "counters": counters,
            "janus": janus_counters,
            "cfg": cfg,
            "stage_pools": pools,
        }

    async def _handle_drain_status(self, msg: dict[str, Any]) -> None:
        """Aggregate loop-local and per-core snapshots without shared reads."""

        rpc_id = msg["rpc_id"]
        timeout = float(msg.get("timeout", 5.0))
        if not 0 < timeout <= 60:
            await self.rpc_async_queue.put(
                {
                    "type": "drain_status_result",
                    "rpc_id": rpc_id,
                    "schema_version": 1,
                    "healthy": False,
                    "drained": False,
                    "stable": False,
                    "replicas": [],
                    "invalid_reasons": ["invalid_timeout"],
                    "errors": [
                        "drain status timeout must be in (0, 60] seconds"
                    ],
                }
            )
            return

        before = self._local_drain_status()
        replica_tasks = [
            pool.get_replica_drain_status(replica_id, timeout=timeout)
            for pool in self.stage_pools
            for replica_id in range(pool.num_replicas)
        ]
        try:
            replicas = await asyncio.wait_for(
                asyncio.gather(*replica_tasks),
                timeout=timeout,
            )
            error = None
        except Exception as exc:
            logger.exception("[Orchestrator] drain status RPC failed")
            replicas = []
            error = f"{type(exc).__name__}: {exc}"

        after = self._local_drain_status()
        stable = before["epoch"] == after["epoch"]
        replicas_healthy = bool(replicas) and all(
            bool(replica.get("healthy")) for replica in replicas
        )
        replicas_drained = bool(replicas) and all(
            bool(replica.get("drained")) for replica in replicas
        )
        # Snapshot churn is normal while the server is busy.  It invalidates a
        # positive drained result, but is not itself a service-health failure.
        healthy = (
            error is None
            and bool(before["healthy"])
            and bool(after["healthy"])
            and replicas_healthy
        )
        drained = (
            healthy
            and stable
            and bool(before["drained"])
            and bool(after["drained"])
            and replicas_drained
        )
        invalid_reasons = [] if stable else ["orchestrator_snapshot_changed"]
        errors = [] if error is None else [error]
        response = {
            "type": "drain_status_result",
            "rpc_id": rpc_id,
            "schema_version": 1,
            "healthy": healthy,
            "drained": drained,
            "stable": stable,
            "epoch": {
                "before": before["epoch"],
                "after": after["epoch"],
            },
            "before": before,
            "after": after,
            "replicas": replicas,
            "invalid_reasons": invalid_reasons,
            "errors": errors,
        }
        await self.rpc_async_queue.put(response)

    # ---- Orchestration loop ----

    async def _orchestration_output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _orchestration_output_handler cancelled")
            return

    async def _orchestration_loop(self) -> None:
        """Poll stage pools and route logical outputs."""
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                pool = self.stage_pools[stage_id]
                for replica_id in range(pool.num_replicas):
                    if self._shutdown_event.is_set():
                        return

                    if pool.stage_type == "diffusion":
                        with nvtx_range("orchestrator:poll_diffusion_output"):
                            output = pool.poll_diffusion_output(replica_id)
                        if output is None:
                            continue
                        self._begin_drain_output()
                        try:
                            await self._handle_processed_outputs(
                                stage_id, replica_id, [output]
                            )
                        finally:
                            self._end_drain_output()
                        idle = False
                    else:
                        try:
                            with nvtx_range("orchestrator:poll_llm_output"):
                                did_work = await self._drain_llm_replica_raw_outputs(
                                    stage_id,
                                    replica_id,
                                    pool,
                                )
                        except asyncio.CancelledError:
                            raise
                        except EngineDeadError as e:
                            logger.error(
                                "[Orchestrator] Stage-%s is dead: %s",
                                stage_id,
                                e,
                            )
                            # TODO: Fault handling is intentionally fail-stop at
                            # the orchestrator level today. If one replica in a
                            # logical stage dies, we promote it to `_fatal_error`,
                            # notify requests already admitted to that stage, and
                            # re-raise so `run()` shuts down all stages. This is
                            # conservative but means a single unhealthy replica in
                            # a multi-replica deployment can take down otherwise
                            # healthy replicas in other stages. Revisit this when
                            # adding per-replica fault isolation / eviction.
                            self._fatal_error = str(e)
                            self._fatal_error_stage_id = stage_id
                            for req_id, req_state in list(self.request_states.items()):
                                if stage_id in req_state.stage_submit_ts:
                                    await self.output_async_queue.put(
                                        {
                                            "type": "error",
                                            "error": str(e),
                                            "fatal": True,
                                            "request_id": req_id,
                                            "stage_id": stage_id,
                                        }
                                    )
                                    self.request_states.pop(req_id, None)
                            self._bump_drain_epoch()
                            self._shutdown_event.set()
                            raise
                        except Exception:
                            if self._shutdown_event.is_set():
                                return
                            logger.exception(
                                "[Orchestrator] Stage-%s replica-%s processing failed",
                                stage_id,
                                replica_id,
                            )
                            raise

                        if did_work:
                            idle = False

            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _drain_llm_replica_raw_outputs(
        self,
        stage_id: int,
        replica_id: int,
        pool: StagePool,
    ) -> bool:
        """Drain ready raw outputs from one LLM replica without blocking."""
        did_work = False
        for _ in range(self._MAX_RAW_OUTPUTS_PER_DRAIN):
            if self._shutdown_event.is_set():
                break

            raw_outputs = pool.poll_llm_raw_output_nowait(replica_id)
            if raw_outputs is None:
                break
            await self._handle_stage_preemptions(
                stage_id,
                list(getattr(raw_outputs, "stage_preempted_request_ids", []) or []),
                dict(getattr(raw_outputs, "stage_preempted_connector_req_ids", {}) or {}),
                dict(getattr(raw_outputs, "stage_preemption_causes", {}) or {}),
                dict(getattr(raw_outputs, "stage_preemption_progress", {}) or {}),
                dict(getattr(raw_outputs, "stage_preemption_rule_ids", {}) or {}),
                dict(
                    getattr(
                        raw_outputs,
                        "stage_preemption_recovery_modes",
                        {},
                    )
                    or {}
                ),
            )
            if raw_outputs.outputs:
                kept_outputs = []
                for eco in raw_outputs.outputs:
                    physical_req_id = getattr(eco, "request_id", None)
                    drop_reason = self._classify_stage_output(
                        stage_id,
                        physical_req_id,
                        float(raw_outputs.timestamp),
                    )
                    if drop_reason is not None:
                        emit_stage_event(
                            "stage_output_version_rejected",
                            stage_id=stage_id,
                            request_id=physical_req_id,
                            drop_reason=drop_reason.value,
                        )
                        continue
                    kept_outputs.append(eco)
                raw_outputs.outputs = kept_outputs
            if not raw_outputs.outputs:
                continue
            await self._submit_next_restart_stage_after_upstream_output(
                stage_id, raw_outputs.outputs
            )
            self._begin_drain_output()
            try:
                if stage_id == 2:
                    self._trace_s2_raw_outputs_polled(raw_outputs)

                await self._handle_kv_ready_raw_outputs(stage_id, raw_outputs)
                physical_to_logical = getattr(
                    self, "_physical_to_logical_request_ids", {}
                )
                for eco in raw_outputs.outputs:
                    physical_req_id = getattr(eco, "request_id", None)
                    logical_req_id = physical_to_logical.get(
                        physical_req_id, physical_req_id
                    )
                    req_state = self.request_states.get(logical_req_id)
                    if req_state is None:
                        continue
                    req_state.streaming.segment_finished = bool(getattr(eco, "is_segment_finished", False))
                    req_state.streaming.new_prompt_len_snapshot = getattr(
                        eco,
                        "new_prompt_len_snapshot",
                        None,
                    )
                # now = _time.monotonic()
                # record_stats = (
                #     self._stat_logger is not None and now - self._last_stats_ts >= self._stats_interval_s
                # )
                record_stats = self._stat_logger is not None
                iteration_stats = IterationStats() if record_stats else None
                if stage_id == 2:
                    with nvtx_range(
                        f"TTFP:s2_raw_output_to_output_processor:replica={replica_id}",
                        color="purple",
                    ):
                        raw_output = await pool.process_llm_raw_outputs(
                            replica_id,
                            raw_outputs,
                            iteration_stats=iteration_stats,
                        )
                else:
                    raw_output = await pool.process_llm_raw_outputs(
                        replica_id,
                        raw_outputs,
                        iteration_stats=iteration_stats,
                    )
                if record_stats:
                    # self._last_stats_ts = now
                    self._stat_logger.record(
                        raw_outputs.scheduler_stats,
                        iteration_stats,
                        engine_idx=self._stage_replica_to_engine_idx[(stage_id, replica_id)],
                    )

                await self._handle_processed_outputs(stage_id, replica_id, raw_output)
                did_work = True
            finally:
                self._end_drain_output()

        return did_work

    async def _handle_processed_outputs(self, stage_id: int, replica_id: int, outputs: list[Any]) -> None:
        """Route processed stage outputs produced by one stage poll."""
        pool = self.stage_pools[stage_id]
        for output in outputs:
            req_state = self.request_states.get(output.request_id)
            if req_state is None:
                logger.warning(
                    "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                    output.request_id,
                    stage_id,
                    list(self.request_states.keys()),
                )
                continue

            if getattr(output, "error", None) is not None:
                await self._handle_stage_error(stage_id, output)
                continue

            stage_metrics = None
            if output.finished:
                stage_metrics = pool.build_stage_metrics(
                    [output],
                    submit_ts=req_state.stage_submit_ts.get(stage_id, _time.time()),
                    replica_id=replica_id,
                )
                stage_metrics.pipeline_timings = dict(req_state.pipeline_timings)

            await self._route_output(stage_id, replica_id, output, req_state, stage_metrics)

    def _classify_stage_output(
        self,
        stage_id: int,
        physical_request_id: str,
        event_monotonic_s: float,
    ) -> MessageDropReason | None:
        """Return why a raw stage output is stale, or ``None`` if current."""

        physical_to_logical = getattr(
            self,
            "_physical_to_logical_request_ids",
            {},
        )
        logical_request_id = physical_to_logical.get(
            physical_request_id,
            physical_request_id,
        )
        req_state = self.request_states.get(logical_request_id)
        if req_state is None or req_state.execution_ledger is None:
            return MessageDropReason.UNKNOWN_LOGICAL_REQUEST
        execution_ledger = req_state.execution_ledger
        current_stage_ref = execution_ledger.current_stages.get(stage_id)
        if current_stage_ref is None:
            return MessageDropReason.INVALID_STAGE
        return execution_ledger.classify_message(
            logical_request_id=logical_request_id,
            pipeline_attempt_id=execution_ledger.pipeline_attempt_id,
            stage_id=stage_id,
            stage_attempt_id=current_stage_ref.stage_attempt_id,
            physical_request_id=physical_request_id,
            event_monotonic_s=event_monotonic_s,
        )

    async def _handle_stage_error(self, stage_id: int, output: Any) -> None:
        """Emit a frontend-visible error and clean up request state."""
        if self._cfg_tracker.is_companion(output.request_id):
            parent_id = self._cfg_tracker.get_parent_id(output.request_id) or output.request_id
        else:
            parent_id = output.request_id
        await self.output_async_queue.put(
            {
                "type": "error",
                "request_id": parent_id,
                "stage_id": stage_id,
                "error": output.error,
            }
        )
        await self._cleanup_request_ids(
            [parent_id, *self._cfg_tracker.cleanup_parent(parent_id)],
            abort=True,
        )

    async def _handle_stage_preemptions(
        self,
        stage_id: int,
        request_ids: list[str],
        connector_req_ids: dict[str, str] | None = None,
        preemption_causes: dict[str, str] | None = None,
        preemption_progress: dict[str, int] | None = None,
        preemption_rule_ids: dict[str, str] | None = None,
        recovery_modes: dict[str, str] | None = None,
    ) -> None:
        """Dispatch each preemption to its explicitly selected recovery path."""
        if not request_ids:
            return

        for physical_request_id in request_ids:
            physical_to_logical = getattr(self, "_physical_to_logical_request_ids", {})
            request_id = physical_to_logical.get(physical_request_id, physical_request_id)
            req_state = self.request_states.get(request_id)
            if req_state is None:
                continue
            execution_ledger = req_state.execution_ledger
            current_stage_ref = (
                execution_ledger.current_stages.get(stage_id)
                if execution_ledger is not None
                else None
            )
            current_stage_request_id = (
                current_stage_ref.physical_request_id
                if current_stage_ref is not None
                else req_state.stage_request_ids.get(stage_id, request_id)
            )
            if physical_request_id != current_stage_request_id:
                # A delayed preemption envelope from a retired execution.
                continue

            requested_mode = (recovery_modes or {}).get(
                physical_request_id,
                PreemptionRecoveryMode.FULL_PIPELINE_RESTART.value,
            )
            if (
                requested_mode
                == PreemptionRecoveryMode.STAGE_LOCAL_RECOMPUTE.value
                and stage_id == 1
                and stage_id < req_state.final_stage_id
                and execution_ledger is not None
            ):
                cutoff = _time.monotonic()
                preemption_cause = (preemption_causes or {}).get(
                    physical_request_id, "allocation_failure"
                )
                discarded_progress = int(
                    (preemption_progress or {}).get(physical_request_id, 0)
                )
                controlled_rule_id = (preemption_rule_ids or {}).get(
                    physical_request_id
                )
                connector_req_id = (connector_req_ids or {}).get(
                    physical_request_id,
                    physical_request_id,
                )
                final_stage_id = req_state.final_stage_id
                affected_request_ids = {stage_id: physical_request_id}
                for downstream_stage_id in range(stage_id + 1, final_stage_id + 1):
                    next_attempt = (
                        execution_ledger.current_stages[
                            downstream_stage_id
                        ].stage_attempt_id
                        + 1
                    )
                    affected_request_ids[downstream_stage_id] = (
                        f"{request_id}__pipeline"
                        f"{execution_ledger.pipeline_attempt_id}"
                        f"__stage{downstream_stage_id}"
                        f"__attempt{next_attempt}"
                    )

                retired = execution_ledger.begin_stage_recompute(
                    stage_id=stage_id,
                    physical_request_ids=affected_request_ids,
                    started_monotonic_s=cutoff,
                )
                req_state.stage_request_ids = {
                    pipeline_stage_id: ref.physical_request_id
                    for pipeline_stage_id, ref in (
                        execution_ledger.current_stages.items()
                    )
                }
                req_state.execution_attempt = execution_ledger.pipeline_attempt_id
                req_state.recovery_target_stage_id = stage_id
                req_state.recovery_target_progress = discarded_progress
                req_state.recovery_preemption_cause = preemption_cause
                req_state.recovery_controlled_rule_id = controlled_rule_id

                qwen_bridge_states = req_state.streaming.bridge_states.get(
                    "qwen3_omni"
                )
                if isinstance(qwen_bridge_states, dict):
                    qwen_bridge_states.pop(request_id, None)

                for affected_stage_id in range(stage_id, final_stage_id + 1):
                    req_state.discard_stage_outputs_before[affected_stage_id] = cutoff
                replay_stage_ids = [
                    downstream_stage_id
                    for downstream_stage_id in range(
                        stage_id + 1, final_stage_id + 1
                    )
                    if self.stage_pools[downstream_stage_id].final_output
                ]
                req_state.begin_output_replay(replay_stage_ids)
                req_state.processed_terminal_stages.difference_update(
                    range(stage_id, final_stage_id + 1)
                )
                for downstream_stage_id in range(stage_id + 1, final_stage_id + 1):
                    retired_request_id = retired[
                        downstream_stage_id
                    ].physical_request_id
                    downstream_pool = self.stage_pools[downstream_stage_id]
                    await downstream_pool.abort_requests([retired_request_id])
                    downstream_pool.release_binding(retired_request_id)
                    req_state.stage_submit_ts.pop(downstream_stage_id, None)
                    replacement_id = affected_request_ids[downstream_stage_id]
                    physical_to_logical[replacement_id] = request_id

                if self.async_chunk:
                    await self._prewarm_async_chunk_stages(
                        request_id,
                        req_state.stage0_request_template,
                        req_state,
                        start_stage_id=stage_id + 1,
                        recovery=True,
                        connector_req_id=connector_req_id,
                    )

                emit_stage_event(
                    "stage_local_recompute_reset",
                    stage_id=stage_id,
                    request_id=request_id,
                    pipeline_attempt_id=execution_ledger.pipeline_attempt_id,
                    stage_attempt_id=(
                        execution_ledger.current_stages[stage_id].stage_attempt_id
                    ),
                    downstream_stage_ids=list(
                        range(stage_id + 1, final_stage_id + 1)
                    ),
                    replay_outputs_to_drop=dict(req_state.replay_outputs_to_drop),
                    preemption_cause=preemption_cause,
                    controlled_rule_id=controlled_rule_id,
                    discarded_num_computed_tokens=discarded_progress,
                )
                self._bump_drain_epoch()
                continue

            next_execution_attempt = (
                execution_ledger.pipeline_attempt_id + 1
                if execution_ledger is not None
                else req_state.execution_attempt + 1
            )
            old_execution_id = req_state.current_execution_id or physical_request_id
            new_execution_id = f"{request_id}__execution{next_execution_attempt}"
            cutoff = _time.monotonic()
            preemption_cause = (preemption_causes or {}).get(
                physical_request_id, "allocation_failure"
            )
            discarded_progress = int(
                (preemption_progress or {}).get(physical_request_id, 0)
            )
            controlled_rule_id = (preemption_rule_ids or {}).get(
                physical_request_id
            )

            final_stage_id = req_state.final_stage_id
            replay_stage_ids = [
                pipeline_stage_id
                for pipeline_stage_id in range(final_stage_id + 1)
                if self.stage_pools[pipeline_stage_id].final_output
            ]
            req_state.begin_output_replay(replay_stage_ids)
            req_state.processed_terminal_stages.clear()
            for pipeline_stage_id in range(final_stage_id + 1):
                req_state.discard_stage_outputs_before[pipeline_stage_id] = cutoff

            old_stage_request_ids = list(
                dict.fromkeys(
                    (
                        ref.physical_request_id
                        for ref in execution_ledger.current_stages.values()
                    )
                    if execution_ledger is not None
                    else req_state.stage_request_ids.values()
                )
            )
            await self._abort_request_ids(old_stage_request_ids)
            self._release_request_bindings(old_stage_request_ids)

            # All runtime-derived bridge and stage progress belongs to the old
            # execution.  Only the immutable user input, parameters and
            # already-emitted frontend count survive.
            req_state.streaming.bridge_states.clear()
            req_state.streaming.segment_finished = False
            req_state.streaming.new_prompt_len_snapshot = None
            req_state.pd_prefill_multimodal_output = None
            req_state.stage_submit_ts.clear()
            next_stage_request_ids = {
                pipeline_stage_id: new_execution_id
                for pipeline_stage_id in range(final_stage_id + 1)
            }
            if execution_ledger is not None:
                execution_ledger.begin_full_restart(
                    physical_request_ids=next_stage_request_ids,
                    started_monotonic_s=cutoff,
                )
                req_state.execution_attempt = execution_ledger.pipeline_attempt_id
                req_state.stage_request_ids = {
                    pipeline_stage_id: ref.physical_request_id
                    for pipeline_stage_id, ref in (
                        execution_ledger.current_stages.items()
                    )
                }
            else:
                req_state.execution_attempt = next_execution_attempt
                req_state.stage_request_ids = next_stage_request_ids
            req_state.current_execution_id = new_execution_id
            req_state.recovery_target_stage_id = stage_id
            req_state.recovery_target_progress = discarded_progress
            req_state.recovery_preemption_cause = preemption_cause
            req_state.recovery_controlled_rule_id = controlled_rule_id
            physical_to_logical[new_execution_id] = request_id

            stage0_request = copy.deepcopy(req_state.stage0_request_template)
            set_execution_global_request_id(stage0_request, new_execution_id)
            stage0_request = merge_pipeline_attempt_metadata_into_request_input(
                stage0_request,
                logical_request_id=request_id,
                execution_id=new_execution_id,
                pipeline_attempt_id=req_state.execution_attempt,
                stage_attempt_id=0,
                recovery_target_stage_id=stage_id,
                recovery_target_num_computed_tokens=discarded_progress,
                recovery_preemption_cause=preemption_cause,
                controlled_rule_id=controlled_rule_id,
            )
            stage0_request.request_id = new_execution_id
            stage0_request.external_req_id = request_id
            stage0_request.connector_req_id = new_execution_id
            stage0_request.output_connector_req_id = new_execution_id
            req_state.current_stage0_request = stage0_request
            req_state.pending_restart_stages = set(range(1, final_stage_id + 1))
            req_state.stage_submit_ts[0] = _time.time()
            await self.stage_pools[0].submit_initial(
                new_execution_id,
                req_state,
                stage0_request,
                prompt_text=req_state.stage0_prompt_text,
            )

            emit_stage_event(
                "full_pipeline_restarted",
                stage_id=stage_id,
                request_id=request_id,
                old_execution_id=old_execution_id,
                new_execution_id=new_execution_id,
                restarted_stage_ids=list(range(final_stage_id + 1)),
                replay_outputs_to_drop=dict(req_state.replay_outputs_to_drop),
                preemption_cause=preemption_cause,
                controlled_rule_id=controlled_rule_id,
                discarded_num_computed_tokens=discarded_progress,
            )
            self._bump_drain_epoch()

    async def _submit_next_restart_stage_after_upstream_output(
        self,
        stage_id: int,
        outputs: list[Any],
    ) -> None:
        """Start one restarted downstream stage after its upstream is live.

        ``abort_requests_async`` only acknowledges that an abort command was
        sent.  Submitting every replacement stage immediately can therefore
        overlap old GPU/model state with the replacement execution.  Waiting
        for an output from the replacement upstream stage gives each stage a
        causal hand-off and preserves true Stage-0 -> Stage-1 -> Stage-2
        re-execution.
        """
        if not self.async_chunk:
            return

        next_stage_id = stage_id + 1
        for output in outputs:
            physical_request_id = getattr(output, "request_id", None)
            physical_to_logical = getattr(
                self, "_physical_to_logical_request_ids", {}
            )
            request_id = physical_to_logical.get(
                physical_request_id, physical_request_id
            )
            req_state = self.request_states.get(request_id)
            if req_state is None:
                continue
            if next_stage_id not in req_state.pending_restart_stages:
                continue
            if physical_request_id != req_state.stage_request_ids.get(stage_id):
                continue

            # Remove before awaiting so another ready output cannot submit the
            # same physical request twice.  Restore it if submission fails.
            req_state.pending_restart_stages.remove(next_stage_id)
            try:
                await self._prewarm_async_chunk_stages(
                    request_id,
                    req_state.current_stage0_request,
                    req_state,
                    start_stage_id=next_stage_id,
                    end_stage_id=next_stage_id,
                    recovery=True,
                    connector_req_id=req_state.current_execution_id,
                )
            except Exception:
                req_state.pending_restart_stages.add(next_stage_id)
                raise

            emit_stage_event(
                "full_pipeline_restart_stage_submitted",
                stage_id=next_stage_id,
                request_id=request_id,
                execution_id=req_state.current_execution_id,
                upstream_stage_id=stage_id,
            )

    # ---- Shared helpers ----

    async def _cleanup_request_ids(self, request_ids: list[str], *, abort: bool = False) -> None:
        """Release pool bindings and logical request state for the given ids."""
        if not request_ids:
            return

        physical_request_ids = list(request_ids)
        physical_to_logical = getattr(
            self, "_physical_to_logical_request_ids", {}
        )
        logical_request_ids = {
            physical_to_logical.get(request_id, request_id)
            for request_id in request_ids
        }
        for logical_request_id in logical_request_ids:
            req_state = self.request_states.get(logical_request_id)
            if req_state is not None and req_state.execution_ledger is not None:
                physical_request_ids.extend(
                    req_state.execution_ledger.cleanup_physical_request_ids()
                )
        for physical_id, logical_id in physical_to_logical.items():
            if logical_id in logical_request_ids:
                physical_request_ids.append(physical_id)
        physical_request_ids = list(dict.fromkeys(physical_request_ids))

        has_recovery_attempts = any(
            physical_id not in logical_request_ids
            for physical_id in physical_request_ids
        )
        if abort or has_recovery_attempts:
            await self._abort_request_ids(physical_request_ids)
        # No further output for these logical requests can be accepted. Clean
        # remaining upstream processor state even if its final envelope has
        # not yet arrived; dropping that late envelope must not leak state.
        for pool in self.stage_pools:
            pool.discard_terminal_output_states(physical_request_ids)
        self._release_request_bindings(physical_request_ids)
        for request_id in logical_request_ids:
            self._pd_kv_params.pop(request_id, None)
            getattr(self, "_omni_nvtx_s2_raw_output_seen", set()).discard(request_id)
            getattr(self, "_omni_nvtx_s2_route_seen", set()).discard(request_id)
            for physical_id, logical_id in list(physical_to_logical.items()):
                if logical_id == request_id:
                    physical_to_logical.pop(physical_id, None)
            running_counter = getattr(self, "_running_counter", None)
            if self.request_states.pop(request_id, None) is not None and running_counter is not None:
                running_counter.decrement()
        self._bump_drain_epoch()

    def _ttfp_submit_range_name(
        self,
        request_id: str,
        next_stage_id: int,
        final_stage_id: int,
    ) -> str | None:
        short_req_id = str(request_id)[-8:]
        if next_stage_id == 1 and final_stage_id > next_stage_id:
            return f"TTFP:s1_submit:req={short_req_id}"
        if next_stage_id == 2:
            return f"TTFP:s2_submit:req={short_req_id}"
        return None

    def _trace_s2_raw_outputs_polled(self, raw_outputs: EngineCoreOutputs) -> None:
        for eco in getattr(raw_outputs, "outputs", []) or []:
            if getattr(eco, "pooling_output", None) is None:
                continue
            req_id = str(getattr(eco, "request_id", ""))
            if not req_id or req_id in self._omni_nvtx_s2_raw_output_seen:
                continue
            self._omni_nvtx_s2_raw_output_seen.add(req_id)
            short_req_id = req_id[-8:]
            nvtx_mark(
                f"TTFP:s2_raw_output_polled:req={short_req_id}",
                color="purple",
            )

    def _maybe_clone_diffusion_params_for_cfg(self, request_id: str, params: Any) -> Any:
        """Attach CFG companion ids to diffusion sampling params when needed."""
        companion_request_ids = self._cfg_tracker.get_companion_request_ids(request_id)
        if not companion_request_ids:
            return params

        import copy

        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if not isinstance(params, OmniDiffusionSamplingParams):
            return params

        params = copy.deepcopy(params)
        params.cfg_kv_request_ids = companion_request_ids
        return params

    async def _route_output(
        self,
        stage_id: int,
        replica_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to frontend and/or forward."""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)

        if finished and self._cfg_tracker.is_companion(req_id):
            await self._handle_cfg_companion_ready(req_id)
            await self._cleanup_request_ids([req_id])
            return

        if finished:
            req_state.processed_terminal_stages.add(stage_id)
            if stage_id == req_state.final_stage_id and not req_state.outputs_drained():
                req_state.completion_deferred = True
                emit_stage_event(
                    "request_completion_deferred",
                    stage_id=stage_id,
                    request_id=req_id,
                    pending_stages=sorted(set(req_state.stage_submit_ts)
                                          - req_state.processed_terminal_stages),
                )

        if self.stage_pools[stage_id].final_output:
            output_sequence = req_state.output_attempt_sequences.get(stage_id, 0)
            output_decision = req_state.classify_and_commit_output(
                stage_id,
                finished=bool(finished),
            )
            if output_decision is OutputDecision.COMMIT:
                emit_stage_event(
                    "stage_output_committed",
                    stage_id=stage_id,
                    request_id=req_id,
                    **req_state.trace_identity_fields(stage_id),
                    output_stream_id=req_state._output_stream_id(stage_id),
                    output_sequence=output_sequence,
                    finished=bool(finished),
                )
            terminal_only_output = None
            if output_decision in {
                OutputDecision.DROP_REPLAY_DUPLICATE,
                OutputDecision.COMMIT_TERMINAL_ONLY,
            }:
                emit_stage_event(
                    "stage_replay_output_suppressed",
                    stage_id=stage_id,
                    request_id=req_id,
                    remaining=req_state.replay_outputs_to_drop[stage_id],
                )
                if output_decision is OutputDecision.COMMIT_TERMINAL_ONLY:
                    # The replay may finish before reproducing every output that
                    # the retired execution already sent.  Suppressing its final
                    # output in that case also suppresses the frontend completion
                    # signal, leaving the HTTP stream open after backend cleanup.
                    # Forward a content-free output so the frontend observes only
                    # termination, without replaying an already-emitted chunk.
                    terminal_only_output = OmniRequestOutput(
                        request_id=req_id,
                        finished=True,
                        stage_id=getattr(output, "stage_id", stage_id),
                        final_output_type=getattr(
                            output,
                            "final_output_type",
                            getattr(
                                self.stage_pools[stage_id],
                                "final_output_type",
                                "text",
                            ),
                        ),
                    )
                    emit_stage_event(
                        "stage_replay_terminal_forwarded",
                        stage_id=stage_id,
                        request_id=req_id,
                        suppressed_outputs_remaining=(
                            req_state.replay_outputs_to_drop[stage_id]
                        ),
                    )
            elif output_decision is not OutputDecision.COMMIT:
                emit_stage_event(
                    "stage_output_commit_rejected",
                    stage_id=stage_id,
                    request_id=req_id,
                    output_decision=output_decision.value,
                )
            queue_payload = {
                "type": "output",
                "request_id": req_id,
                "stage_id": stage_id,
                "replica_id": replica_id,
                "engine_outputs": terminal_only_output or output,
                "metrics": stage_metrics,
                "finished": finished and stage_id == req_state.final_stage_id
                and req_state.outputs_drained(),
                "stage_submit_ts": submit_ts,
            }
            if queue_payload["finished"] and (
                terminal_only_output is not None or output_decision is OutputDecision.COMMIT
            ):
                req_state.completion_deferred = False
            if terminal_only_output is not None:
                await self.output_async_queue.put(queue_payload)
            elif output_decision is not OutputDecision.COMMIT:
                pass
            elif stage_id == 2 and req_id not in self._omni_nvtx_s2_route_seen:
                self._omni_nvtx_s2_route_seen.add(req_id)
                short_req_id = str(req_id)[-8:]
                with nvtx_range(
                    f"TTFP:s2_output_processor_to_route:req={short_req_id}",
                    color="blue",
                ):
                    nvtx_mark(
                        f"TTFP:s2_route_final_output:req={short_req_id}:finished={int(bool(finished))}",
                        color="blue",
                    )
                with nvtx_range(
                    f"TTFP:s2_route_to_frontend_dispatch:req={short_req_id}",
                    color="blue",
                ):
                    await self.output_async_queue.put(queue_payload)
            else:
                await self.output_async_queue.put(queue_payload)
        elif stage_metrics is not None:
            await self.output_async_queue.put(
                {
                    "type": "stage_metrics",
                    "request_id": req_id,
                    "stage_id": stage_id,
                    "replica_id": replica_id,
                    "metrics": stage_metrics,
                    "stage_submit_ts": submit_ts,
                }
            )

        if self._pd_pair is not None and finished and stage_id == self._pd_pair[0]:
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is not None:
                self._pd_kv_params[req_id] = kv_params if isinstance(kv_params, dict) else dict(kv_params)
            req_state.pd_prefill_multimodal_output = getattr(output, "multimodal_output", None)

        if (
            (finished or (req_state.streaming.enabled and req_state.streaming.segment_finished))
            and stage_id < req_state.final_stage_id
            and not self.async_chunk
            and (not self._next_stage_already_submitted(stage_id, req_state) or req_state.streaming.enabled)
        ):
            if (
                finished
                and self._cfg_tracker.has_companions(req_id)
                and not self._cfg_tracker.all_companions_done(req_id)
            ):
                self._cfg_tracker.defer_parent(req_id, output, stage_id)
            else:
                await self._forward_to_next_stage(
                    req_id,
                    stage_id,
                    output,
                    req_state,
                    is_streaming_session=req_state.streaming.enabled,
                    is_final_update=False,
                )
                if req_state.streaming.enabled and finished:
                    # For streaming sessions, send the terminal (resumable=False) update only on a finish
                    await self._forward_to_next_stage(
                        req_id,
                        stage_id,
                        output,
                        req_state,
                        is_streaming_session=True,
                        is_final_update=True,
                    )

        if finished and req_state.outputs_drained():
            if req_state.completion_deferred:
                # FIFO control envelope, after every required output. No fake
                # audio/text chunk and no holding back the real final audio.
                await self.output_async_queue.put({
                    "type": "request_done",
                    "request_id": req_id,
                    "stage_id": req_state.final_stage_id,
                    "finished": True,
                })
                req_state.completion_deferred = False
                emit_stage_event(
                    "request_completion_released",
                    stage_id=req_state.final_stage_id,
                    request_id=req_id,
                    processed_terminal_stages=sorted(req_state.processed_terminal_stages),
                )
            if req_state.execution_attempt > 0:
                emit_stage_event(
                    "full_pipeline_recovery_completed",
                    stage_id=req_state.final_stage_id,
                    request_id=req_state.request_id,
                    execution_id=req_state.current_execution_id,
                    pipeline_attempt_id=req_state.execution_attempt,
                    recovery_target_stage_id=(
                        req_state.recovery_target_stage_id
                    ),
                    recovery_target_num_computed_tokens=(
                        req_state.recovery_target_progress
                    ),
                    preemption_cause=req_state.recovery_preemption_cause,
                    controlled_rule_id=(
                        req_state.recovery_controlled_rule_id
                    ),
                )
            await self._cleanup_request_ids([req_id, *self._cfg_tracker.cleanup_parent(req_id)])

    def _next_stage_already_submitted(self, stage_id: int, req_state: OrchestratorRequestState) -> bool:
        return (stage_id + 1) in req_state.stage_submit_ts

    async def _handle_cfg_companion_ready(self, req_id: str) -> None:
        """Mark a CFG companion as done; if all companions are done, flush deferred parent."""
        parent_id = self._cfg_tracker.on_companion_completed(req_id)
        if parent_id is None:
            return

        deferred = self._cfg_tracker.pop_pending_parent(parent_id)
        if deferred is None:
            return

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            return

        stage_id = deferred["stage_id"]
        if (stage_id + 1) in parent_state.stage_submit_ts:
            return

        await self._forward_to_next_stage(
            parent_id,
            stage_id,
            deferred["engine_outputs"],
            parent_state,
        )

    async def _handle_kv_ready_raw_outputs(
        self,
        stage_id: int,
        raw_outputs: EngineCoreOutputs,
    ) -> None:
        """Forward split requests once stage-0 KV is ready."""
        if self.async_chunk:
            return

        for raw_output in raw_outputs.outputs:
            kv_params = getattr(raw_output, "kv_transfer_params", None)
            if not (isinstance(kv_params, dict) and kv_params.get("kv_ready")):
                continue

            req_id = raw_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue
            if self._cfg_tracker.is_companion(req_id):
                await self._handle_cfg_companion_ready(req_id)
                continue
            if stage_id >= req_state.final_stage_id:
                continue
            if (stage_id + 1) in req_state.stage_submit_ts:
                continue

            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, raw_output, stage_id)
            else:
                await self._forward_to_next_stage(req_id, stage_id, raw_output, req_state)

    def _build_pd_decode_params(self, req_id: str, sp: Any) -> Any:
        """Build decode-side sampling params with KV transfer params for PD routing.

        Clones the sampling params and injects kv_transfer_params that tell the
        decode engine where to pull the KV cache from (prefill engine's bootstrap addr).
        """
        sp = sp.clone()
        if sp.extra_args is None:
            sp.extra_args = {}

        # Get KV params captured from the prefill output (must include remote_request_id).
        kv_prefill_params = self._pd_kv_params.pop(req_id, None)
        if not kv_prefill_params or "remote_request_id" not in kv_prefill_params:
            raise RuntimeError(
                f"[Orchestrator][PD] Missing prefill kv_transfer_params.remote_request_id for req={req_id}"
            )

        decode_kv_params: dict[str, Any] = {
            "transfer_id": f"xfer-{req_id}",
        }

        if self._pd_bootstrap_addr:
            decode_kv_params["remote_bootstrap_addr"] = self._pd_bootstrap_addr

        if self._pd_prefill_engine_id:
            decode_kv_params["remote_engine_id"] = self._pd_prefill_engine_id

        # Overlay params from prefill side (includes remote_request_id set by monkey patch).
        decode_kv_params.update(kv_prefill_params)

        # Ensure these flags are set correctly after any overlay.
        decode_kv_params["do_remote_prefill"] = True
        decode_kv_params["do_remote_decode"] = False
        if not decode_kv_params.get("transfer_id"):
            decode_kv_params["transfer_id"] = f"xfer-{req_id}"

        sp.extra_args["kv_transfer_params"] = decode_kv_params

        logger.debug(
            "[Orchestrator][PD] decode kv_transfer_params for req=%s: %s",
            req_id,
            decode_kv_params,
        )
        return sp

    async def _forward_to_next_stage(
        self,
        req_id: str,
        src_stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        is_streaming_session: bool = False,
        is_final_update: bool = False,
    ) -> None:
        """Forward output from the current logical stage to the next one."""
        next_logical = src_stage_id + 1
        next_stage_request_id = req_state.stage_request_ids.get(next_logical, req_id)
        next_pool = self.stage_pools[next_logical]
        next_client = next_pool.stage_client
        params = req_state.sampling_params_list[next_logical]
        source_outputs = [output]
        next_stage_resumable = is_streaming_session and not is_final_update
        already_submitted = self._next_stage_already_submitted(src_stage_id, req_state)
        requires_multimodal_data = getattr(next_client, "requires_multimodal_data", False)

        if next_pool.stage_type == "diffusion":
            if next_client.custom_process_input_func is not None:
                _t_ar2d = _time.perf_counter()
                diffusion_prompt = next_client.custom_process_input_func(
                    source_outputs,
                    req_state.prompt,
                    requires_multimodal_data,
                )
                _dt_ar2d = (_time.perf_counter() - _t_ar2d) * 1000
                req_state.pipeline_timings["ar2diffusion_ms"] = _dt_ar2d
                logger.info(
                    "[Orchestrator] ar2diffusion req=%s wall_time=%.3fms stage=%d->%d",
                    req_id,
                    _dt_ar2d,
                    src_stage_id,
                    next_logical,
                )
                if isinstance(diffusion_prompt, list):
                    if not diffusion_prompt:
                        error_output = OmniRequestOutput.from_error(
                            req_id,
                            f"Stage-{src_stage_id} produced no valid inputs for diffusion stage-{next_logical}",
                        )
                        logger.warning(
                            "[Orchestrator] req=%s stage=%d produced empty diffusion inputs for stage=%d; "
                            "routing terminal error output",
                            req_id,
                            src_stage_id,
                            next_logical,
                        )
                        await self.output_async_queue.put(
                            {
                                "type": "output",
                                "request_id": req_id,
                                "stage_id": next_logical,
                                "engine_outputs": error_output,
                                "metrics": None,
                                "finished": True,
                            }
                        )
                        await self._cleanup_request_ids(
                            [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                        )
                        return
                    if already_submitted and len(diffusion_prompt) == 1:
                        diffusion_prompt = diffusion_prompt[0]
            else:
                diffusion_prompt = req_state.prompt

            diffusion_prompt = merge_scheduling_metadata_into_request_input(
                diffusion_prompt,
                req_state.scheduling_metadata,
            )

            if already_submitted:
                await next_pool.submit_update(next_stage_request_id, req_state, diffusion_prompt)
            else:
                ttfp_range_name = self._ttfp_submit_range_name(
                    req_id,
                    next_logical,
                    req_state.final_stage_id,
                )
                if ttfp_range_name is not None:
                    with nvtx_range(ttfp_range_name, color="magenta"):
                        await next_pool.submit_initial(
                            next_stage_request_id,
                            req_state,
                            diffusion_prompt,
                            submit_kwargs={
                                "kv_sender_info": self._build_kv_sender_info(
                                    list(getattr(next_client, "engine_input_source", None) or [src_stage_id]),
                                    request_id=next_stage_request_id,
                                )
                            },
                            params_override=self._maybe_clone_diffusion_params_for_cfg(req_id, params),
                        )
                else:
                    await next_pool.submit_initial(
                        next_stage_request_id,
                        req_state,
                        diffusion_prompt,
                        submit_kwargs={
                            "kv_sender_info": self._build_kv_sender_info(
                                list(getattr(next_client, "engine_input_source", None) or [src_stage_id]),
                                request_id=next_stage_request_id,
                            )
                        },
                        params_override=self._maybe_clone_diffusion_params_for_cfg(req_id, params),
                    )
            req_state.stage_submit_ts[next_logical] = _time.time()
            return

        # PD disaggregation: prefill → decode routing uses original prompt + KV transfer params
        if self._pd_pair is not None and (src_stage_id, next_logical) == self._pd_pair:
            params = self._build_pd_decode_params(req_id, params)

            # Use the original user prompt for the decode stage (not processed embeddings)
            original_prompt = req_state.prompt
            raw_decode_inputs = [original_prompt] if not isinstance(original_prompt, list) else original_prompt

            decode_inputs: list[dict[str, Any]] = []
            for decode_input in raw_decode_inputs:
                if isinstance(decode_input, dict):
                    decode_inputs.append(decode_input)
                    continue
                prompt_token_ids = getattr(decode_input, "prompt_token_ids", None)
                if prompt_token_ids is None:
                    raise TypeError(
                        "[Orchestrator][PD] decode input must be dict or have prompt_token_ids, "
                        f"got {type(decode_input).__name__} for req={req_id}"
                    )
                decode_inputs.append({"prompt_token_ids": list(prompt_token_ids)})

            for decode_input in decode_inputs:
                decode_input = merge_scheduling_metadata_into_prompt(
                    decode_input,
                    req_state.scheduling_metadata,
                )
                request = build_engine_core_request_from_tokens(
                    request_id=next_stage_request_id,
                    prompt=decode_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                    mm_features=req_state.mm_features,
                    resumable=next_stage_resumable,
                )
                request.external_req_id = req_id
                request.connector_req_id = next_stage_request_id
                request.output_connector_req_id = next_stage_request_id
                if already_submitted:
                    await next_pool.submit_update(next_stage_request_id, req_state, request)
                else:
                    ttfp_range_name = self._ttfp_submit_range_name(
                        req_id,
                        next_logical,
                        req_state.final_stage_id,
                    )
                    if ttfp_range_name is not None:
                        with nvtx_range(ttfp_range_name, color="magenta"):
                            await next_pool.submit_initial(
                                next_stage_request_id, req_state, request, prompt_text=None
                            )
                    else:
                        await next_pool.submit_initial(
                            next_stage_request_id, req_state, request, prompt_text=None
                        )

            req_state.stage_submit_ts[next_logical] = _time.time()
            return

        if req_state.pd_prefill_multimodal_output is not None:
            req_state.streaming.bridge_states.setdefault(
                "pd_prefill_multimodal_output_by_req",
                {},
            )[req_id] = req_state.pd_prefill_multimodal_output

        try:
            with stage_trace_span(
                "stage_route_to_next",
                stage_id=src_stage_id,
                request_id=next_stage_request_id,
                nvtx_name=f"s{src_stage_id}_route_to_s{next_logical}",
                nvtx_color="purple",
                to_stage=next_logical,
                already_submitted=already_submitted,
                streaming=is_streaming_session,
                final_update=is_final_update,
            ):
                next_inputs = next_client.process_engine_inputs(
                    source_outputs,
                    req_state.prompt,
                    streaming_context=req_state.streaming,
                )
        except Exception:
            logger.exception(
                "[Orchestrator] req=%s process_engine_inputs FAILED for stage-%s",
                req_id,
                next_logical,
            )
            raise

        # Build and submit requests for each input
        for next_input in next_inputs:
            next_input = merge_scheduling_metadata_into_request_input(
                next_input,
                req_state.scheduling_metadata,
            )
            # Only AR thinker stages consume encoder mm_features; downstream
            # (talker/code2wav/…) must not see them (avoids encoder-cache misses).
            model_stage = getattr(next_client, "model_stage", None)
            mm_features = req_state.mm_features if model_stage == "thinker" else None
            request = build_engine_core_request_from_tokens(
                request_id=next_stage_request_id,
                prompt=next_input,
                params=params,
                model_config=next_pool.stage_vllm_config.model_config,
                mm_features=mm_features,
                resumable=next_stage_resumable,
            )

            request.external_req_id = req_id
            request.connector_req_id = next_stage_request_id
            request.output_connector_req_id = next_stage_request_id
            if already_submitted:
                with nvtx_range("orchestrator:submit_update"):
                    await next_pool.submit_update(next_stage_request_id, req_state, request)
            else:
                ttfp_range_name = self._ttfp_submit_range_name(
                    req_id,
                    next_logical,
                    req_state.final_stage_id,
                )
                with nvtx_range("orchestrator:submit_initial"):
                    if ttfp_range_name is not None:
                        with nvtx_range(ttfp_range_name, color="magenta"):
                            await next_pool.submit_initial(
                                next_stage_request_id, req_state, request, prompt_text=None
                            )
                    else:
                        await next_pool.submit_initial(
                            next_stage_request_id, req_state, request, prompt_text=None
                        )

        req_state.stage_request_ids[next_logical] = next_stage_request_id
        physical_to_logical = getattr(
            self, "_physical_to_logical_request_ids", None
        )
        if physical_to_logical is not None:
            physical_to_logical[next_stage_request_id] = req_id
        req_state.stage_submit_ts[next_logical] = _time.time()

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
        *,
        start_stage_id: int = 1,
        end_stage_id: int | None = None,
        recovery: bool = False,
        connector_req_id: str | None = None,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode."""
        if req_state.final_stage_id <= 0:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        last_stage_id = (
            req_state.final_stage_id if end_stage_id is None else end_stage_id
        )
        for next_stage_id in range(start_stage_id, last_stage_id + 1):
            stage_request_id = req_state.stage_request_ids.get(next_stage_id, request_id)
            next_pool = self.stage_pools[next_stage_id]
            params = req_state.sampling_params_list[next_stage_id]

            req_state.stage_submit_ts[next_stage_id] = _time.time()

            if next_pool.stage_type == "diffusion":
                prewarm_prompt = merge_scheduling_metadata_into_request_input(
                    req_state.prompt,
                    req_state.scheduling_metadata,
                )
                await next_pool.submit_initial(
                    stage_request_id,
                    req_state,
                    prewarm_prompt,
                    submit_kwargs={
                        "kv_sender_info": self._build_kv_sender_info(
                            list(getattr(next_pool.stage_client, "engine_input_source", None) or [next_stage_id - 1]),
                            request_id=stage_request_id,
                        )
                    },
                )
            else:
                from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length

                try:
                    next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
                except Exception:
                    next_prompt_len = max(1, len(prompt_token_ids))

                original_prompt = req_state.prompt
                if isinstance(original_prompt, dict):
                    base_input = copy.deepcopy(original_prompt)
                else:
                    base_input = {}

                if recovery:
                    set_execution_global_request_id(
                        base_input,
                        connector_req_id or stage_request_id,
                    )
                    base_input = merge_pipeline_attempt_metadata_into_request_input(
                        base_input,
                        logical_request_id=request_id,
                        execution_id=(
                            req_state.current_execution_id or request_id
                        ),
                        pipeline_attempt_id=req_state.execution_attempt,
                        stage_attempt_id=(
                            req_state.execution_ledger.current_stages[
                                next_stage_id
                            ].stage_attempt_id
                            if req_state.execution_ledger is not None
                            else 0
                        ),
                        recovery_target_stage_id=(
                            req_state.recovery_target_stage_id
                            if req_state.recovery_target_stage_id is not None
                            else next_stage_id
                        ),
                        recovery_target_num_computed_tokens=(
                            req_state.recovery_target_progress or 0
                        ),
                        recovery_preemption_cause=(
                            req_state.recovery_preemption_cause
                            or "allocation_failure"
                        ),
                        controlled_rule_id=(
                            req_state.recovery_controlled_rule_id
                        ),
                    )

                # A recovery consumer must wait for replayed connector data.
                # Reusing the ordinary prewarm placeholder makes Stage-2 run
                # Talker's prompt length as if it were codec tokens before the
                # first replay chunk arrives.
                # Stage 1 needs the ordinary fixed-length placeholder so its
                # first scheduling step is non-empty.  Later stages must wait
                # for newly generated upstream chunks from this execution.
                wait_for_replayed_upstream = recovery and next_stage_id > 1
                base_input["prompt_token_ids"] = (
                    [] if wait_for_replayed_upstream else [0] * next_prompt_len
                )
                base_input["multi_modal_data"] = None
                base_input["mm_processor_kwargs"] = None
                base_input = merge_scheduling_metadata_into_prompt(
                    base_input,
                    req_state.scheduling_metadata,
                )

                request = build_engine_core_request_from_tokens(
                    request_id=stage_request_id,
                    prompt=base_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                    connector_req_id=connector_req_id or request_id,
                    output_connector_req_id=connector_req_id or request_id,
                )
                request.external_req_id = request_id
                await next_pool.submit_initial(stage_request_id, req_state, request, prompt_text=None)
            req_state.stage_request_ids[next_stage_id] = stage_request_id
            physical_to_logical = getattr(
                self, "_physical_to_logical_request_ids", None
            )
            if physical_to_logical is not None:
                physical_to_logical[stage_request_id] = request_id

    def _build_kv_sender_info(
        self,
        sender_stage_ids: list[int],
        *,
        request_id: str | None = None,
    ) -> dict[int, dict[str, Any]] | None:
        """Build per-request sender info for diffusion KV-transfer receivers."""
        sender_infos: dict[int, dict[str, Any]] = {}
        for sender_stage_id in dict.fromkeys(sender_stage_ids):
            if sender_stage_id < 0 or sender_stage_id >= len(self.stage_pools):
                continue

            sender_pool = self.stage_pools[sender_stage_id]
            sender_stage = sender_pool.get_bound_client(request_id) if request_id is not None else None
            if sender_stage is None:
                sender_stage = sender_pool.stage_client
            get_sender_info = getattr(sender_stage, "get_kv_sender_info", None)
            if not callable(get_sender_info):
                continue

            sender_info = get_sender_info()
            if not sender_info:
                logger.warning(
                    "[Orchestrator] Stage-%s has no KV sender info available",
                    sender_stage_id,
                )
                continue

            sender_infos[sender_stage_id] = sender_info

        return sender_infos or None

    # ---- Shutdown / lifecycle ----

    async def _drain_pending_requests_on_fatal(self) -> None:
        """Drain the request queue and broadcast fatal errors for any
        pending add_request messages that were never processed.

        Called from the ``run()`` finally block when a fatal error
        (e.g. ``EngineDeadError``) caused the orchestrator to shut down
        before the request handler could process all queued messages.
        Also broadcasts for any already-tracked requests still in
        ``request_states`` that were not yet notified.
        """
        assert self._fatal_error is not None

        notified: set[str] = set()

        # 1) Drain pending messages from the request queue.
        while True:
            try:
                msg = self.request_async_queue.get_nowait()
            except Exception:
                break
            if msg.get("type") == "add_request":
                req_id = msg["request_id"]
                await self.output_async_queue.put(
                    {
                        "type": "error",
                        "error": self._fatal_error,
                        "fatal": True,
                        "request_id": req_id,
                        "stage_id": self._fatal_error_stage_id,
                    }
                )
                notified.add(req_id)

        # 2) Broadcast for any tracked requests not already notified
        #    (e.g. request was registered but the EngineDeadError handler
        #    missed it because it wasn't submitted to the dead stage yet).
        for req_id in list(self.request_states):
            if req_id not in notified:
                await self.output_async_queue.put(
                    {
                        "type": "error",
                        "error": self._fatal_error,
                        "fatal": True,
                        "request_id": req_id,
                        "stage_id": self._fatal_error_stage_id,
                    }
                )
            self.request_states.pop(req_id, None)

    def _shutdown_stages(self) -> None:
        """Shutdown all stage pools."""
        if self._stages_shutdown:
            return

        self._stages_shutdown = True
        total = sum(pool.num_replicas for pool in self.stage_pools)
        logger.info("[Orchestrator] Shutting down all %d client(s)", total)
        for pool in self.stage_pools:
            for replica_id in range(pool.num_replicas):
                pool.shutdown_replica(replica_id)
