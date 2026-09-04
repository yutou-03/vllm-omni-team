"""Scheduling contracts shared by benchmark and serving components."""

from vllm_omni.scheduling.admission import (
    AdmissionAction,
    AdmissionControllerConfig,
    AdmissionControllerState,
    AdmissionDecision,
    StaticConcurrencyEnvelope,
    decide_admission_cap,
    replay_admission_trace,
)

from vllm_omni.scheduling.contracts import (
    RequestPath,
    RequestSLOContract,
    SLOMilestone,
    StageRuntimeSnapshot,
)
from vllm_omni.scheduling.deadline import (
    MilestoneCostEstimate,
    StageDeadlineSnapshot,
    active_stage_milestones,
    derive_stage_deadline_snapshot,
)
from vllm_omni.scheduling.cost_model import StaticMilestoneCostModel
from vllm_omni.scheduling.config import StageDDLConfig

from vllm_omni.scheduling.metadata import (
    CLIENT_SCHEDULING_FIELD,
    CURRENT_SCHEDULING_SCHEMA_VERSION,
    LEGACY_SCHEDULING_SCHEMA_VERSION,
    SCHEDULING_SCHEMA_VERSION,
    bind_server_scheduling_metadata,
    build_client_scheduling_metadata,
    build_server_scheduling_metadata,
    derive_stage_deadlines,
    extract_scheduling_metadata,
    merge_scheduling_metadata_into_additional_information,
    merge_scheduling_metadata_into_prompt,
    normalize_client_scheduling_metadata,
)
from vllm_omni.scheduling.policy import (
    BASELINE_POLICY_ENV,
    BaselineSchedulingPolicy,
    SchedulingMetadataError,
    get_baseline_scheduling_policy,
    policy_applies_to_stage,
    policy_key,
    remaining_prefill_tokens,
)
from vllm_omni.scheduling.request_queue import (
    PolicyOrderedRequestQueue,
    maybe_create_policy_ordered_queue,
)

__all__ = [
    "AdmissionAction",
    "AdmissionControllerConfig",
    "AdmissionControllerState",
    "AdmissionDecision",
    "CLIENT_SCHEDULING_FIELD",
    "CURRENT_SCHEDULING_SCHEMA_VERSION",
    "LEGACY_SCHEDULING_SCHEMA_VERSION",
    "RequestPath",
    "RequestSLOContract",
    "SLOMilestone",
    "StageRuntimeSnapshot",
    "MilestoneCostEstimate",
    "StageDeadlineSnapshot",
    "StaticMilestoneCostModel",
    "StageDDLConfig",
    "StaticConcurrencyEnvelope",
    "active_stage_milestones",
    "SCHEDULING_SCHEMA_VERSION",
    "BASELINE_POLICY_ENV",
    "BaselineSchedulingPolicy",
    "PolicyOrderedRequestQueue",
    "SchedulingMetadataError",
    "build_client_scheduling_metadata",
    "bind_server_scheduling_metadata",
    "build_server_scheduling_metadata",
    "derive_stage_deadlines",
    "derive_stage_deadline_snapshot",
    "decide_admission_cap",
    "extract_scheduling_metadata",
    "get_baseline_scheduling_policy",
    "merge_scheduling_metadata_into_additional_information",
    "merge_scheduling_metadata_into_prompt",
    "normalize_client_scheduling_metadata",
    "maybe_create_policy_ordered_queue",
    "policy_applies_to_stage",
    "policy_key",
    "remaining_prefill_tokens",
    "replay_admission_trace",
]
