"""Small NVTX helpers for optional Nsight Systems profiling.

The helpers are intentionally no-op unless VLLM_NVTX_SCOPES_FOR_PROFILING=1.
This keeps profiling annotations from affecting normal serving paths and makes
call sites safe on non-CUDA platforms.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - profiling must be optional.
    torch = None  # type: ignore[assignment]

try:
    import nvtx as _nvtx
except Exception:  # pragma: no cover - optional dependency.
    _nvtx = None

_ENV_FLAG = "VLLM_NVTX_SCOPES_FOR_PROFILING"
_ACTIVE_RANGES: dict[str, Any] = {}
_ACTIVE_RANGES_LOCK = Lock()


def _enabled() -> bool:
    return os.getenv(_ENV_FLAG, "").lower() in {"1", "true", "yes", "on"}


def _cuda_nvtx_enabled() -> bool:
    return bool(_enabled() and torch is not None and torch.cuda.is_available())


def _decorate_name(name: str, color: str | None = None) -> str:
    # torch.cuda.nvtx does not expose colors; keep a color hint in the name so
    # ranges are still easy to filter when the optional nvtx package is absent.
    return f"{name} [color={color}]" if color else name


def nvtx_mark(name: str, *, color: str | None = None) -> None:
    """Emit an NVTX mark when profiling annotations are enabled."""
    if not _enabled():
        return
    if _nvtx is not None:
        try:
            _nvtx.mark(message=name, color=color)
            return
        except Exception:
            pass
    if not _cuda_nvtx_enabled():
        return
    try:
        torch.cuda.nvtx.mark(_decorate_name(name, color))
    except Exception:
        # NVTX annotations must never break inference.
        return


@contextmanager
def nvtx_range(name: str, *, color: str | None = None) -> Iterator[None]:
    """Emit an NVTX range when profiling annotations are enabled."""
    pushed_kind: str | None = None
    if _enabled():
        if _nvtx is not None:
            try:
                _nvtx.push_range(message=name, color=color)
                pushed_kind = "nvtx"
            except Exception:
                pushed_kind = None
        if pushed_kind is None and _cuda_nvtx_enabled():
            try:
                torch.cuda.nvtx.range_push(_decorate_name(name, color))
                pushed_kind = "torch"
            except Exception:
                pushed_kind = None
    try:
        yield
    finally:
        if pushed_kind == "nvtx" and _nvtx is not None:
            try:
                _nvtx.pop_range()
            except Exception:
                return
        elif pushed_kind == "torch":
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                return


def nvtx_start_range(name: str, *, color: str | None = None) -> Any | None:
    """Start a non-stack NVTX range when supported, otherwise emit a start mark."""
    if not _enabled():
        return None
    if _nvtx is not None:
        try:
            return ("nvtx", _nvtx.start_range(message=name, color=color))
        except Exception:
            pass
    if _cuda_nvtx_enabled() and hasattr(torch.cuda.nvtx, "range_start"):
        try:
            return ("torch", torch.cuda.nvtx.range_start(_decorate_name(name, color)))
        except Exception:
            pass
    nvtx_mark(f"{name}:start", color=color)
    return ("mark", _decorate_name(name, color), color)


def nvtx_end_range(handle: Any | None, name: str | None = None, *, color: str | None = None) -> None:
    """End a non-stack NVTX range, or emit an end mark for fallback handles."""
    if handle is None:
        return
    kind = handle[0] if isinstance(handle, tuple) and handle else None
    try:
        if kind == "nvtx" and _nvtx is not None:
            _nvtx.end_range(handle[1])
        elif kind == "torch" and _cuda_nvtx_enabled() and hasattr(torch.cuda.nvtx, "range_end"):
            torch.cuda.nvtx.range_end(handle[1])
        elif kind == "mark":
            nvtx_mark(f"{name or handle[1]}:end", color=color or handle[2])
    except Exception:
        return


def nvtx_start_keyed_range(key: str, name: str, *, color: str | None = None) -> None:
    """Start a range identified by key if it is not already active."""
    with _ACTIVE_RANGES_LOCK:
        if key in _ACTIVE_RANGES:
            return
        _ACTIVE_RANGES[key] = nvtx_start_range(name, color=color)


def nvtx_end_keyed_range(key: str, name: str | None = None, *, color: str | None = None) -> None:
    """End and remove a keyed range if it is active."""
    with _ACTIVE_RANGES_LOCK:
        handle = _ACTIVE_RANGES.pop(key, None)
    if handle is None:
        if name is not None:
            nvtx_mark(f"{name}:end_without_start", color=color)
        return
    nvtx_end_range(handle, name, color=color)
