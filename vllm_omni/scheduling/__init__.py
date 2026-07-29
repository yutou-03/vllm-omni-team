"""Scheduling contracts shared by benchmark and serving components."""

from vllm_omni.scheduling.metadata import (
    CLIENT_SCHEDULING_FIELD,
    SCHEDULING_SCHEMA_VERSION,
    build_client_scheduling_metadata,
    build_server_scheduling_metadata,
    derive_stage_deadlines,
    extract_scheduling_metadata,
    merge_scheduling_metadata_into_additional_information,
    merge_scheduling_metadata_into_prompt,
    normalize_client_scheduling_metadata,
)
from vllm_omni.scheduling.policy import (
    ACTIVE_PREEMPTION_MAX_PER_REQUEST_ENV,
    ACTIVE_PREEMPTION_MAX_RECOMPUTE_TOKENS_ENV,
    ACTIVE_PREEMPTION_MIN_DEADLINE_GAIN_MS_ENV,
    BASELINE_POLICY_ENV,
    BaselineSchedulingPolicy,
    SchedulingMetadataError,
    get_active_preemption_config,
    get_baseline_scheduling_policy,
    policy_applies_to_stage,
    policy_is_deadline_edf,
    policy_key,
    policy_uses_active_preemption,
    remaining_prefill_tokens,
)
from vllm_omni.scheduling.request_queue import (
    PolicyOrderedRequestQueue,
    maybe_create_policy_ordered_queue,
)

__all__ = [
    "CLIENT_SCHEDULING_FIELD",
    "SCHEDULING_SCHEMA_VERSION",
    "ACTIVE_PREEMPTION_MAX_PER_REQUEST_ENV",
    "ACTIVE_PREEMPTION_MAX_RECOMPUTE_TOKENS_ENV",
    "ACTIVE_PREEMPTION_MIN_DEADLINE_GAIN_MS_ENV",
    "BASELINE_POLICY_ENV",
    "BaselineSchedulingPolicy",
    "PolicyOrderedRequestQueue",
    "SchedulingMetadataError",
    "build_client_scheduling_metadata",
    "build_server_scheduling_metadata",
    "derive_stage_deadlines",
    "extract_scheduling_metadata",
    "get_active_preemption_config",
    "get_baseline_scheduling_policy",
    "merge_scheduling_metadata_into_additional_information",
    "merge_scheduling_metadata_into_prompt",
    "normalize_client_scheduling_metadata",
    "maybe_create_policy_ordered_queue",
    "policy_applies_to_stage",
    "policy_is_deadline_edf",
    "policy_key",
    "policy_uses_active_preemption",
    "remaining_prefill_tokens",
]
