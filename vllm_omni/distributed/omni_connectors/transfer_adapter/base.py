# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import deque
from typing import Any

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniTransferAdapterBase:
    """Base class for managing data transfer via OmniConnector.

    This class handles the core loop logic and connector interactions, but
    leaves the specific data processing (chunks, KV cache, etc.) to subclasses.
    """

    def __init__(self, config: Any):
        self.config = config
        if not hasattr(self, "connector"):
            self.connector = None
        # Requests that are waiting to be polled
        self._pending_load_reqs = deque()
        # Requests that have successfully retrieved data
        self._finished_load_reqs = set()
        self._cancelled_load_reqs: set[str] = set()

        # Requests that are waiting to be saved
        self._pending_save_reqs = deque()
        # Requests that have successfully saved data
        self._finished_save_reqs = set()

        # Drain snapshots are requested from the EngineCore main thread while
        # recv/save work runs in these two background threads.  Queue length by
        # itself is insufficient: a worker can have popped the last item and
        # still be processing it.  Keep dequeue + in-flight transitions under
        # one lock so a zero snapshot cannot observe that gap.
        self._drain_state_lock = threading.Lock()
        self._recv_inflight = 0
        self._save_inflight = 0
        # A save exception loses the dequeued task and is therefore not a
        # transient poll miss.  Persist the first such fault so later empty
        # queues cannot be reported as healthy/drained.
        self._background_fault: str | None = None

        self.stop_event = threading.Event()
        self._recv_cond = threading.Condition()
        self._save_cond = threading.Condition()

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()

    @classmethod
    def create_connector(cls, model_config: Any):
        raise NotImplementedError

    def recv_loop(self):
        """Loop to poll for incoming data.

        Process each pending request exactly once per pass.  When no request
        made progress, back off 1 ms instead of tight-spinning on failed
        shm_open syscalls (which can burn a full CPU core).
        """
        while not self.stop_event.is_set():
            with self._drain_state_lock:
                n = len(self._pending_load_reqs)
            any_success = False
            for _ in range(n):
                with self._drain_state_lock:
                    if not self._pending_load_reqs:
                        break
                    request = self._pending_load_reqs.popleft()
                    self._recv_inflight += 1
                try:
                    request_id = request.request_id
                    if request_id in self._cancelled_load_reqs:
                        self._cancelled_load_reqs.discard(request_id)
                        continue
                    self.request_ids_mapping[request_id] = request.external_req_id
                    is_success = self._poll_single_request(request)
                    if is_success:
                        any_success = True
                    else:
                        with self._drain_state_lock:
                            self._pending_load_reqs.append(request)
                except Exception as e:
                    with self._drain_state_lock:
                        self._pending_load_reqs.append(request)
                    self._record_background_fault("recv", e)
                    try:
                        failed_request_id = getattr(
                            request, "request_id", "<unknown>"
                        )
                    except Exception:
                        failed_request_id = "<unreadable>"
                    logger.warning(
                        "Error receiving data for %s: %s",
                        failed_request_id,
                        e,
                    )
                finally:
                    with self._drain_state_lock:
                        self._recv_inflight -= 1

            # Timeout is the fallback for lock-free append/notify races.
            with self._recv_cond:
                with self._drain_state_lock:
                    pending_loads = len(self._pending_load_reqs)
                if not pending_loads and not self.stop_event.is_set():
                    self._recv_cond.wait(timeout=0.1)
                elif not any_success and not self.stop_event.is_set():
                    self._recv_cond.wait(timeout=0.001)

    def save_loop(self):
        """Loop to send outgoing data."""
        while not self.stop_event.is_set():
            while True:
                with self._drain_state_lock:
                    if not self._pending_save_reqs:
                        break
                    task = self._pending_save_reqs.popleft()
                    self._save_inflight += 1
                try:
                    self._send_single_request(task)
                except Exception as e:
                    self._record_background_fault("save", e)
                    try:
                        request_id = (
                            task.get("request_id", "<unknown>")
                            if isinstance(task, dict)
                            else getattr(task, "request_id", "<unknown>")
                        )
                    except Exception:
                        request_id = "<unreadable>"
                    logger.warning("Error saving data for %s: %s", request_id, e)
                finally:
                    with self._drain_state_lock:
                        self._save_inflight -= 1

            with self._save_cond:
                with self._drain_state_lock:
                    pending_saves = len(self._pending_save_reqs)
                if not pending_saves and not self.stop_event.is_set():
                    self._save_cond.wait(timeout=0.1)

    def _enqueue_load_request(self, request: Any) -> None:
        with self._drain_state_lock:
            self._pending_load_reqs.append(request)

    def _enqueue_save_request(self, task: Any) -> None:
        with self._drain_state_lock:
            self._pending_save_reqs.append(task)

    def _record_background_fault(
        self,
        worker: str,
        error: BaseException,
    ) -> None:
        summary = f"{worker}: {type(error).__name__}: {error}"
        with self._drain_state_lock:
            if self._background_fault is None:
                self._background_fault = summary

    def get_drain_status(self) -> dict[str, Any]:
        """Return a read-only background-transfer snapshot.

        The lock makes ``pending == inflight == 0`` a strict boundary: neither
        background thread can be between dequeue and in-flight accounting.
        Subclasses may add scheduler-owned bookkeeping, but must not mutate it.
        """

        with self._drain_state_lock:
            counters = {
                "pending_load_requests": len(self._pending_load_reqs),
                "pending_save_requests": len(self._pending_save_reqs),
                "recv_inflight": self._recv_inflight,
                "save_inflight": self._save_inflight,
            }
            background_fault = self._background_fault
        workers_alive = {
            "recv": bool(
                getattr(self, "recv_thread", None) is not None
                and self.recv_thread.is_alive()
            ),
            "save": bool(
                getattr(self, "save_thread", None) is not None
                and self.save_thread.is_alive()
            ),
        }
        healthy = (
            not self.stop_event.is_set()
            and all(workers_alive.values())
            and background_fault is None
        )
        status = {
            "healthy": healthy,
            "drained": healthy and all(value == 0 for value in counters.values()),
            "workers_alive": workers_alive,
            "counters": counters,
        }
        errors = []
        if background_fault is not None:
            errors.append(background_fault)
        dead_workers = [name for name, alive in workers_alive.items() if not alive]
        if dead_workers:
            errors.append(
                "background transfer worker not alive: "
                + ", ".join(dead_workers)
            )
        if self.stop_event.is_set():
            errors.append("transfer adapter is stopped")
        if errors:
            status["error"] = "; ".join(errors)
        return status

    def _poll_single_request(self, *args, **kwargs):
        """Poll connector for a single request task.
        Subclasses should implement request-specific receive behavior."""
        raise NotImplementedError

    def _send_single_request(self, *args, **kwargs):
        """Send one pending save request task to the connector.
        Subclasses should implement task-specific handling logic."""
        raise NotImplementedError

    def load_async(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save_async(self, *args, **kwargs):
        """Submit data to be saved. To be implemented by subclasses."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Load request data from connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Save data to connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        """Get finished loaded or saved requests"""
        raise NotImplementedError

    def shutdown(self):
        """Stop background loops and close the connector."""
        self.stop_event.set()
        with self._recv_cond:
            self._recv_cond.notify_all()
        with self._save_cond:
            self._save_cond.notify_all()
        if self.connector is not None:
            try:
                self.connector.close()
            except Exception:
                pass
