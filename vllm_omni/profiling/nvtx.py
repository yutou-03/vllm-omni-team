"""Small NVTX helpers for optional Nsight Systems profiling.

The helpers are intentionally no-op unless VLLM_NVTX_SCOPES_FOR_PROFILING=1.
This keeps profiling annotations from affecting normal serving paths and makes
call sites safe on non-CUDA platforms.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator

import torch

_ENV_FLAG = "VLLM_NVTX_SCOPES_FOR_PROFILING"


def _enabled() -> bool:
    return os.getenv(_ENV_FLAG, "").lower() in {"1", "true", "yes", "on"} and torch.cuda.is_available()


def nvtx_mark(name: str) -> None:
    """Emit an NVTX mark when profiling annotations are enabled."""
    if not _enabled():
        return
    try:
        torch.cuda.nvtx.mark(name)
    except Exception:
        # NVTX annotations must never break inference.
        return


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range when profiling annotations are enabled."""
    pushed = False
    if _enabled():
        try:
            torch.cuda.nvtx.range_push(name)
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                return
