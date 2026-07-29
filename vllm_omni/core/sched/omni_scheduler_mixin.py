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
    get_active_preemption_config,
    get_baseline_scheduling_policy,
    policy_applies_to_stage,
    policy_is_deadline_edf,
    policy_key,
    policy_uses_active_preemption,
)
from vllm_omni.scheduling.request_queue import (
    PolicyOrderedRequestQueue,
    maybe_create_policy_ordered_queue,
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
        self._baseline_data_ready_order: dict[str, int] = {}
        self._baseline_next_data_ready_order = 0
        self._baseline_conformance_snapshot: dict[str, Any] | None = None
        self._baseline_policy_order_snapshot: dict[str, Any] | None = None
        self._baseline_waiting_queue_choices: list[dict[str, Any]] = []
        self._baseline_preempted_pending_resume: set[str] = set()
        self._baseline_active_preemption_config = get_active_preemption_config()
        self._baseline_active_preemption_evaluation: dict[str, Any] | None = None
        self._baseline_active_preemption_records: list[dict[str, Any]] = []

        if not self._baseline_policy_applies():
            return

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

    def _baseline_mark_data_ready(
        self,
        request_id: str,
        *,
        ready_time: float,
    ) -> None:
        if request_id in self._baseline_data_ready_time:
            return
        ready_order = self._baseline_next_data_ready_order
        self._baseline_next_data_ready_order += 1
        self._baseline_data_ready_time[request_id] = ready_time
        self._baseline_data_ready_order[request_id] = ready_order
        emit_stage_event(
            "stage_data_ready",
            stage_id=self._baseline_stage_id,
            request_id=request_id,
            data_ready_monotonic_s=ready_time,
            stage_ready_order=ready_order,
        )

    def _baseline_validate_and_track_admission(self, request: Request) -> None:
        if self._baseline_request_is_ready_on_add():
            self._baseline_mark_data_ready(
                str(request.request_id),
                ready_time=time.monotonic(),
            )
        if self._baseline_policy_applies():
            self._baseline_request_key(request)

    @staticmethod
    def _baseline_queue_insertion_order(queue: Any) -> list[Request]:
        if isinstance(queue, PolicyOrderedRequestQueue):
            return queue.insertion_order()
        return list(queue)

    @staticmethod
    def _baseline_request_ids(requests: list[Request]) -> list[str]:
        return [str(request.request_id) for request in requests]

    def _baseline_prepare_schedule(
        self,
        *,
        token_budget_before: int | None = None,
    ) -> None:
        """Refresh data-ready times and order runnable requests in place."""

        self._baseline_waiting_queue_choices = []
        self._baseline_active_preemption_evaluation = None
        self._baseline_active_preemption_records = []
        trace_mechanism = conformance_trace_enabled()
        before_orders: dict[str, list[str]] = {}
        if trace_mechanism:
            before_orders = {
                "running": self._baseline_request_ids(list(self.running)),
                "waiting": self._baseline_request_ids(
                    self._baseline_queue_insertion_order(self.waiting)
                ),
                "skipped_waiting": self._baseline_request_ids(
                    self._baseline_queue_insertion_order(self.skipped_waiting)
                ),
            }

        adapter = getattr(self, "chunk_transfer_adapter", None)
        if adapter is not None:
            ready_time = time.monotonic()
            for request_id in adapter.requests_with_ready_chunks:
                self._baseline_mark_data_ready(
                    str(request_id),
                    ready_time=ready_time,
                )
        if self._baseline_policy_applies():
            self.running.sort(key=self._baseline_request_key)

        after_policy_orders = {
            "running": self._baseline_request_ids(list(self.running)),
            "waiting": self._baseline_request_ids(list(self.waiting)),
            "skipped_waiting": self._baseline_request_ids(
                list(self.skipped_waiting)
            ),
        }
        self._baseline_maybe_preempt_running()

        if trace_mechanism:
            after_preemption_orders = {
                "running": self._baseline_request_ids(list(self.running)),
                "waiting": self._baseline_request_ids(list(self.waiting)),
                "skipped_waiting": self._baseline_request_ids(
                    list(self.skipped_waiting)
                ),
            }
            reordered_domains = [
                domain
                for domain in before_orders
                if before_orders[domain] != after_policy_orders[domain]
            ]
            self._baseline_policy_order_snapshot = {
                "mechanism_trace_version": 2,
                "queue_orders_before_policy": before_orders,
                "queue_orders_after_policy": after_policy_orders,
                "queue_orders_after_active_preemption": (
                    after_preemption_orders
                ),
                "policy_reordered": bool(reordered_domains),
                "policy_reordered_domains": reordered_domains,
            }
        else:
            self._baseline_policy_order_snapshot = None
        self._baseline_capture_conformance_snapshot(
            token_budget_before=token_budget_before,
        )

    @staticmethod
    def _baseline_request_can_compete(request: Request) -> bool:
        status_name = getattr(
            getattr(request, "status", None),
            "name",
            str(getattr(request, "status", "")),
        ).lower()
        return not any(
            marker in status_name
            for marker in (
                "finished",
                "waiting_for_input",
                "waiting_for_chunk",
                "waiting_for_remote",
            )
        )

    @staticmethod
    def _baseline_running_can_be_preempted(request: Request) -> bool:
        if not OmniSchedulerMixin._baseline_request_can_compete(request):
            return False
        if (
            request.num_output_placeholders > 0
            and request.max_tokens is not None
            and request.num_computed_tokens
            + 2
            - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        ):
            return False
        return True

    def _baseline_maybe_preempt_running(self) -> None:
        """Measure deadline inversion and optionally replace one running request.

        Only the explicit ``*_edf_p`` policies mutate scheduler state. The
        non-preemptive EDF policies still emit the same evaluation, which makes
        it possible to estimate the opportunity rate before enabling the more
        expensive recompute-preemption mechanism.
        """

        policy = self._baseline_policy
        if not policy_is_deadline_edf(policy):
            return

        running = [
            request
            for request in self.running
            if self._baseline_running_can_be_preempted(request)
        ]
        waiting = [
            request
            for request in list(self.waiting)
            + list(getattr(self, "skipped_waiting", ()))
            if self._baseline_request_can_compete(request)
        ]
        max_running = int(getattr(self, "max_num_running_reqs", len(self.running)))
        full_running = len(self.running) >= max_running
        evaluation: dict[str, Any] = {
            "evaluated": bool(full_running and running and waiting),
            "running_capacity_full": full_running,
            "num_running_candidates": len(running),
            "num_waiting_candidates": len(waiting),
            "deadline_inversion": False,
            "active_preemption_enabled": policy_uses_active_preemption(policy),
            "preempted": False,
        }
        self._baseline_active_preemption_evaluation = evaluation
        if not evaluation["evaluated"]:
            return

        best_waiting = min(waiting, key=self._baseline_request_key)
        worst_running = max(running, key=self._baseline_request_key)
        best_waiting_deadline = float(self._baseline_request_key(best_waiting)[0])
        worst_running_deadline = float(self._baseline_request_key(worst_running)[0])
        deadline_gain_ms = (
            worst_running_deadline - best_waiting_deadline
        ) * 1000.0
        evaluation.update(
            waiting_request_id=str(best_waiting.request_id),
            waiting_deadline_monotonic_s=best_waiting_deadline,
            worst_running_request_id=str(worst_running.request_id),
            worst_running_deadline_monotonic_s=worst_running_deadline,
            deadline_gain_ms=deadline_gain_ms,
            deadline_inversion=deadline_gain_ms > 0.0,
        )
        if deadline_gain_ms <= 0.0 or not policy_uses_active_preemption(policy):
            return

        config = self._baseline_active_preemption_config
        max_recompute_tokens = int(config["max_recompute_tokens"])
        max_per_request = int(config["max_per_request"])
        min_deadline_gain_ms = float(config["min_deadline_gain_ms"])
        feasible_victims = [
            request
            for request in running
            if int(request.num_computed_tokens) <= max_recompute_tokens
            and int(getattr(request, "num_preemptions", 0)) < max_per_request
            and (
                float(self._baseline_request_key(request)[0])
                - best_waiting_deadline
            )
            * 1000.0
            >= min_deadline_gain_ms
            and float(self._baseline_request_key(request)[0])
            > best_waiting_deadline
        ]
        if not feasible_victims:
            if deadline_gain_ms < min_deadline_gain_ms:
                rejection_reason = "deadline_gain_below_guard"
            elif all(
                int(request.num_computed_tokens) > max_recompute_tokens
                for request in running
            ):
                rejection_reason = "recompute_token_cap"
            else:
                rejection_reason = "per_request_preemption_cap"
            evaluation["rejection_reason"] = rejection_reason
            return

        victim = max(feasible_victims, key=self._baseline_request_key)
        victim_deadline = float(self._baseline_request_key(victim)[0])
        victim_computed_tokens = int(victim.num_computed_tokens)
        record = {
            "waiting_request_id": str(best_waiting.request_id),
            "victim_request_id": str(victim.request_id),
            "waiting_deadline_monotonic_s": best_waiting_deadline,
            "victim_deadline_monotonic_s": victim_deadline,
            "deadline_gain_ms": (
                victim_deadline - best_waiting_deadline
            )
            * 1000.0,
            "victim_num_computed_tokens_before": victim_computed_tokens,
            "victim_num_preemptions_before": int(
                getattr(victim, "num_preemptions", 0)
            ),
            "max_recompute_tokens": max_recompute_tokens,
            "max_per_request": max_per_request,
            "min_deadline_gain_ms": min_deadline_gain_ms,
        }
        self.running.remove(victim)
        self._preempt_request(victim, time.monotonic())
        self._baseline_active_preemption_records.append(record)
        evaluation.update(
            preempted=True,
            selected_victim_request_id=str(victim.request_id),
            selected_victim_num_computed_tokens_before=victim_computed_tokens,
        )

    def _baseline_attach_active_preemptions(self, scheduler_output: Any) -> None:
        """Expose pre-schedule active victims through vLLM's normal output."""

        active_ids = {
            record["victim_request_id"]
            for record in self._baseline_active_preemption_records
        }
        if not active_ids:
            return
        preempted_req_ids = getattr(scheduler_output, "preempted_req_ids", None)
        if preempted_req_ids is None:
            scheduler_output.preempted_req_ids = active_ids
        else:
            preempted_req_ids.update(active_ids)

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
                if (
                    location == "running"
                    and request.num_output_placeholders > 0
                    and request.max_tokens is not None
                    and request.num_computed_tokens
                    + 2
                    - request.num_output_placeholders
                    >= request.num_prompt_tokens + request.max_tokens
                ):
                    # Match Scheduler.schedule's async final-placeholder skip:
                    # the previous step has already selected enough work to
                    # reach max_tokens, so this request is only waiting for its
                    # final sampled token to replace the placeholder.  Other
                    # requests selected in the previous step remain runnable.
                    ineligible_reasons[request_id] = (
                        ConformanceIneligibleReason.ASYNC_OUTPUT_PENDING_AT_TOKEN_LIMIT.value
                    )
                    continue
                if location != "running" and sequence_slots == 0:
                    capacity_reason = (
                        ConformanceIneligibleReason.SEQUENCE_SLOT_LIMIT
                        if policy_uses_active_preemption(
                            self._baseline_policy
                        )
                        else ConformanceIneligibleReason.NONPREEMPTIVE_RUNNING_CAPACITY
                    )
                    ineligible_reasons[request_id] = capacity_reason.value
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

        shadow_policy_orders: dict[str, dict[str, list[str]]] = {}
        queue_requests = {
            "running": running,
            "waiting": waiting,
            "skipped_waiting": skipped_waiting,
        }
        before_orders = (self._baseline_policy_order_snapshot or {}).get(
            "queue_orders_before_policy", {}
        )
        runnable_by_location = {
            location: [
                request
                for request in requests
                if str(request.request_id) in runnable_req_ids
            ]
            for location, requests in queue_requests.items()
        }
        for policy in BaselineSchedulingPolicy:
            policy_orders: dict[str, list[str]] = {}
            for location, requests in runnable_by_location.items():
                if policy is BaselineSchedulingPolicy.NATIVE_FCFS:
                    original_order = before_orders.get(location)
                    if original_order is None:
                        original_order = [str(request.request_id) for request in requests]
                    runnable_ids = {str(request.request_id) for request in requests}
                    policy_orders[location] = [
                        request_id
                        for request_id in original_order
                        if request_id in runnable_ids
                    ]
                    continue
                try:
                    ordered = sorted(
                        requests,
                        key=lambda request: policy_key(
                            request,
                            policy=policy,
                            stage_id=self._baseline_stage_id,
                            data_ready_time=self._baseline_request_data_ready_time(
                                request
                            ),
                            now=time.monotonic(),
                        ),
                    )
                except (TypeError, ValueError):
                    # Shadow tracing must never change the live scheduler. A
                    # custom-policy run will independently reject bad metadata.
                    policy_orders[location] = []
                else:
                    policy_orders[location] = [
                        str(request.request_id) for request in ordered
                    ]
            shadow_policy_orders[policy.value] = policy_orders

        self._baseline_conformance_snapshot = {
            "policy": self._baseline_policy.value,
            "policy_applies_to_stage": self._baseline_policy_applies(),
            "policy_key_kind": policy_key_kind,
            "runnable_req_ids": runnable_req_ids,
            "policy_keys": policy_keys,
            "policy_domains": policy_domains,
            "request_locations": request_locations,
            "request_num_computed_tokens_before": {
                request_id: int(request.num_computed_tokens)
                for request in running + waiting + skipped_waiting
                if (request_id := str(request.request_id)) in request_locations
            },
            "ineligible_reasons": ineligible_reasons,
            "token_budget_before": token_budget_before,
            "kv_cache_usage_before": getattr(
                getattr(self, "kv_cache_manager", None),
                "usage",
                None,
            ),
            "sequence_slots_before": sequence_slots,
            "stage_ready_order": {
                request_id: self._baseline_data_ready_order[request_id]
                for request_id in request_locations
                if request_id in self._baseline_data_ready_order
            },
            "shadow_policy_orders": shadow_policy_orders,
            "shadow_nonpreemptive_running_req_ids": [
                str(request.request_id)
                for request in runnable_by_location["running"]
            ],
            "shadow_waiting_blocked_by_nonpreemption_req_ids": sorted(
                request_id
                for request_id, reason in ineligible_reasons.items()
                if reason
                == ConformanceIneligibleReason.NONPREEMPTIVE_RUNNING_CAPACITY.value
            ),
            **(self._baseline_policy_order_snapshot or {}),
        }

    @staticmethod
    def _baseline_shadow_decision_summary(
        conformance: dict[str, Any],
        *,
        scheduled_req_ids: list[str],
    ) -> dict[str, Any]:
        """Compare policy queue prefixes while holding actual admission counts fixed.

        This is deliberately not a second scheduler implementation.  It
        answers the narrower, auditable question needed for diagnosis: at a
        real selection boundary, would a policy substitute waiting requests in
        the same queue if it received the same number of admissions?  Running
        requests are reported as non-preemptive locks rather than candidates
        for replacement.
        """

        locations = conformance.get("request_locations", {})
        orders = conformance.get("shadow_policy_orders", {})
        fcfs_orders = orders.get(BaselineSchedulingPolicy.NATIVE_FCFS.value, {})
        result: dict[str, Any] = {
            "comparison": "fixed_location_admission_count",
            "nonpreemptive_running_req_ids": conformance.get(
                "shadow_nonpreemptive_running_req_ids", []
            ),
            "waiting_blocked_by_nonpreemption_req_ids": conformance.get(
                "shadow_waiting_blocked_by_nonpreemption_req_ids", []
            ),
            "policies": {},
        }
        for policy_name, policy_orders in orders.items():
            locations_summary: dict[str, Any] = {}
            for location in ("waiting", "skipped_waiting"):
                actual_selected = [
                    request_id
                    for request_id in scheduled_req_ids
                    if locations.get(request_id) == location
                ]
                selected_count = len(actual_selected)
                fcfs_order = list(fcfs_orders.get(location, ()))
                shadow_order = list(policy_orders.get(location, ()))
                fcfs_prefix = fcfs_order[:selected_count]
                shadow_prefix = shadow_order[:selected_count]
                locations_summary[location] = {
                    "candidate_order": shadow_order,
                    "actual_selected_req_ids": actual_selected,
                    "selected_count": selected_count,
                    "fcfs_prefix_req_ids": fcfs_prefix,
                    "shadow_prefix_req_ids": shadow_prefix,
                    "order_differs_from_fcfs": shadow_order != fcfs_order,
                    "fixed_count_substitution_from_fcfs": (
                        set(shadow_prefix) != set(fcfs_prefix)
                    ),
                    "fixed_count_substitution_from_actual": (
                        set(shadow_prefix) != set(actual_selected)
                    ),
                }
            result["policies"][policy_name] = locations_summary
        return result

    def _baseline_forget_request_ids(self, request_ids: Any) -> None:
        ready_times = getattr(self, "_baseline_data_ready_time", None)
        if ready_times is None:
            return
        if request_ids is None:
            ready_times.clear()
            self._baseline_data_ready_order.clear()
            return
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        for request_id in request_ids:
            ready_times.pop(request_id, None)
            self._baseline_data_ready_order.pop(request_id, None)

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
            waiting_request = self.waiting.peek_request()
            skipped_request = self.skipped_waiting.peek_request()
            waiting_key = self._baseline_request_key(waiting_request)
            skipped_key = self._baseline_request_key(skipped_request)
            choose_waiting = waiting_key <= skipped_key
            self._baseline_waiting_queue_choices.append(
                {
                    "waiting_head_request_id": str(waiting_request.request_id),
                    "waiting_head_policy_key": list(waiting_key),
                    "skipped_waiting_head_request_id": str(
                        skipped_request.request_id
                    ),
                    "skipped_waiting_head_policy_key": list(skipped_key),
                    "chosen_queue": (
                        "waiting" if choose_waiting else "skipped_waiting"
                    ),
                    # Native FCFS always resumes skipped_waiting first.
                    "differs_from_native_fcfs": choose_waiting,
                }
            )
            return self.waiting if choose_waiting else self.skipped_waiting
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
        preempted_req_ids = {
            str(rid)
            for rid in (
                getattr(scheduler_output, "preempted_req_ids", set()) or set()
            )
        }
        active_preemption_records = list(
            self._baseline_active_preemption_records
        )
        active_preempted_req_ids = {
            str(record["victim_request_id"])
            for record in active_preemption_records
        }
        kv_preempted_req_ids = preempted_req_ids - active_preempted_req_ids
        pending_resume = self._baseline_preempted_pending_resume
        resumed_preempted_req_ids = [
            request_id
            for request_id in scheduled_req_ids
            if request_id in pending_resume
        ]
        pending_resume.difference_update(resumed_preempted_req_ids)
        pending_resume.update(preempted_req_ids)
        queue_orders_after_schedule = {
            "running": self._baseline_request_ids(list(self.running)),
            "waiting": self._baseline_request_ids(
                self._baseline_queue_insertion_order(self.waiting)
            ),
            "skipped_waiting": self._baseline_request_ids(
                self._baseline_queue_insertion_order(self.skipped_waiting)
            ),
        }
        preempted_requeue_order = [
            request_id
            for request_id in queue_orders_after_schedule["waiting"]
            if request_id in preempted_req_ids
        ]
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
        request_locations = conformance.get("request_locations", {})
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

        selection_changed_domains: list[str] = []
        if policy_activation:
            # This is a local mechanism check against the queue order captured
            # at this scheduler tick, not a cross-run FCFS counterfactual.
            before_orders = conformance.get("queue_orders_before_policy", {})
            for location, before_order in before_orders.items():
                eligible_before = [
                    request_id
                    for request_id in before_order
                    if request_id in runnable_set
                    and request_locations.get(request_id) == location
                ]
                selected_here = [
                    request_id
                    for request_id in scheduled_req_ids
                    if request_id in runnable_set
                    and request_locations.get(request_id) == location
                ]
                if not selected_here or len(selected_here) == len(eligible_before):
                    continue
                expected_prefix = eligible_before[: len(selected_here)]
                if set(selected_here) != set(expected_prefix):
                    selection_changed_domains.append(location)
        token_budget_before = conformance.get("token_budget_before")
        token_budget_after = (
            max(int(token_budget_before) - total_num_scheduled_tokens, 0)
            if token_budget_before is not None
            else None
        )
        max_running = int(getattr(self, "max_num_running_reqs", len(self.running)))
        conformance_fields = dict(conformance)
        if conformance:
            computed_tokens_before = conformance.get(
                "request_num_computed_tokens_before",
                {},
            )
            preempted_num_computed_tokens_before = {
                request_id: computed_tokens_before.get(request_id)
                for request_id in preempted_requeue_order
            }
            preempted_num_computed_tokens_before.update(
                {
                    str(record["victim_request_id"]): int(
                        record["victim_num_computed_tokens_before"]
                    )
                    for record in active_preemption_records
                }
            )
            shadow_decision = None
            if policy_activation:
                shadow_decision = self._baseline_shadow_decision_summary(
                    conformance,
                    scheduled_req_ids=scheduled_req_ids,
                )
            conformance_fields.update(
                selected_req_ids=scheduled_req_ids,
                num_scheduled_tokens={
                    str(request_id): int(num_tokens)
                    for request_id, num_tokens in num_scheduled_tokens.items()
                },
                token_budget_after=token_budget_after,
                sequence_slots_after=max(max_running - len(self.running), 0),
                policy_activation=policy_activation,
                unselected_runnable_req_ids=sorted(
                    runnable_set - selected_set
                ),
                selection_differs_from_pre_policy_prefix=bool(
                    selection_changed_domains
                ),
                selection_changed_domains=selection_changed_domains,
                shadow_policy_decision=shadow_decision,
                waiting_queue_choices=list(
                    self._baseline_waiting_queue_choices
                ),
                queue_orders_after_schedule=queue_orders_after_schedule,
                kv_allocation_failed=bool(kv_preempted_req_ids),
                kv_allocation_failure_victim_req_ids=[
                    request_id
                    for request_id in preempted_requeue_order
                    if request_id in kv_preempted_req_ids
                ],
                preempted_num_computed_tokens_before=(
                    preempted_num_computed_tokens_before
                ),
                preempted_requeue_order=preempted_requeue_order,
                resumed_preempted_req_ids=resumed_preempted_req_ids,
                active_preemption_evaluation=(
                    self._baseline_active_preemption_evaluation
                ),
                active_preemption_records=active_preemption_records,
                active_preempted_req_ids=sorted(active_preempted_req_ids),
            )
        conformance_fields.setdefault(
            "active_preemption_evaluation",
            self._baseline_active_preemption_evaluation,
        )
        conformance_fields.setdefault(
            "active_preemption_records",
            active_preemption_records,
        )
        conformance_fields.setdefault(
            "active_preempted_req_ids",
            sorted(active_preempted_req_ids),
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
            preempted_req_ids=sorted(preempted_req_ids),
            kv_cache_usage=getattr(self.kv_cache_manager, "usage", None),
            **conformance_fields,
        )
        self._baseline_conformance_snapshot = None
        self._baseline_policy_order_snapshot = None
        self._baseline_waiting_queue_choices = []
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
            preempted_req_ids=sorted(preempted_req_ids),
            kv_cache_usage=getattr(self.kv_cache_manager, "usage", None),
            kv_allocation_failed=bool(kv_preempted_req_ids),
            kv_allocation_failure_victim_req_ids=[
                request_id
                for request_id in preempted_requeue_order
                if request_id in kv_preempted_req_ids
            ],
            preempted_num_computed_tokens_before=(
                preempted_num_computed_tokens_before
                if conformance
                else {
                    str(record["victim_request_id"]): int(
                        record["victim_num_computed_tokens_before"]
                    )
                    for record in active_preemption_records
                }
            ),
            preempted_requeue_order=preempted_requeue_order,
            resumed_preempted_req_ids=resumed_preempted_req_ids,
            queue_orders_after_schedule=queue_orders_after_schedule,
            active_preemption_evaluation=(
                self._baseline_active_preemption_evaluation
            ),
            active_preemption_records=active_preemption_records,
            active_preempted_req_ids=sorted(active_preempted_req_ids),
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
        self._baseline_active_preemption_evaluation = None
        self._baseline_active_preemption_records = []

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
