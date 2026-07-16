"""Scheduling contracts shared by benchmark and serving components."""

from vllm_omni.scheduling.metadata import (
    CLIENT_SCHEDULING_FIELD,
    SCHEDULING_SCHEMA_VERSION,
    build_client_scheduling_metadata,
    normalize_client_scheduling_metadata,
)

__all__ = [
    "CLIENT_SCHEDULING_FIELD",
    "SCHEDULING_SCHEMA_VERSION",
    "build_client_scheduling_metadata",
    "normalize_client_scheduling_metadata",
]
