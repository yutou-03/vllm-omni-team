# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.profiling.nvtx import nvtx_mark, nvtx_range
from vllm_omni.scheduling.metadata import preserve_scheduling_metadata

from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniTransferAdapterBase

logger = get_connector_logger(__name__)


class OmniChunkTransferAdapter(OmniTransferAdapterBase):
    """Chunk-level transfer adapter for Omni connector pipelines.

    This class coordinates per-request chunk exchange between adjacent stages,
    and implements asynchronous get/put of chunks via background threads.
    It tracks per-request chunk indices for put/get, and accumulates
    payloads across chunks (concatenating tensors/lists in AR mode). It also
    caches prompt token ids and additional information for scheduler use.

    Scheduler integration is handled via WAITING_FOR_CHUNK transitions:
    requests are moved to waiting for chunk deque while polling, then restored
    to waiting/running queues once a chunk arrives. The requests will finish
    loading chunk util detecting the payload "finished" flag.

    The base class owns background recv/save loops; load/save only enqueue
    work and return immediately.
    """

    def __init__(self, vllm_config: Any):
        model_config = vllm_config.model_config
        self.scheduler_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.connector = self.create_connector(model_config)
        super().__init__(model_config)
        self.model_mode = getattr(model_config, "worker_type", None) or "ar"
        # State specific to Chunk management
        self.custom_process_next_stage_input_func = None
        custom_process_next_stage_input_func = getattr(model_config, "custom_process_next_stage_input_func", None)
        if custom_process_next_stage_input_func:
            module_path, func_name = custom_process_next_stage_input_func.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_process_next_stage_input_func = getattr(module, func_name)
        # mapping for request id and chunk id
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.code_prompt_token_ids: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()
        self.requests_origin_status = {}

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load_async(self, request: Request):
        """Register a request for asynchronous chunk retrieval.

        This method does not read from the connector directly. It records
        request metadata and enqueues the request id for the background
        receive loop to poll.

        Stage-0 has no upstream producer, so this call is a no-op there.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        self._cancelled_load_reqs.discard(request.request_id)
        self._enqueue_load_request(request)
        with self._recv_cond:
            self._recv_cond.notify()

    def save_async(
        self,
        pooling_output: torch.Tensor | None = None,
        request: Request | None = None,
    ):
        """Build and enqueue one chunk for asynchronous sending.

        Payload extraction happens in ``_send_single_request`` on the
        background save_loop thread.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
        """
        task = {
            "pooling_output": pooling_output,
            "request": request,
            "is_finished": request.is_finished(),
        }
        if request is not None:
            req_id = getattr(request, "external_req_id", getattr(request, "request_id", ""))
            # Merely tracing an enqueue must not create sender bookkeeping.
            # Some terminal requests (for example text-only requests that end
            # at stage 0) legitimately produce no next-stage payload.
            chunk_id = self.put_req_chunk.get(req_id, 0)
            nvtx_mark(f"s{self.connector.stage_id}_save_async:req={str(req_id)[-8:]}:chunk={chunk_id}")
        self._enqueue_save_request(task)
        with self._save_cond:
            self._save_cond.notify()

    def _poll_single_request(self, request: Request):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        req_id = request.request_id
        chunk_id = self.get_req_chunk[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        try:
            with nvtx_range(f"s{stage_id}_connector_get:req={str(external_req_id)[-8:]}:chunk={chunk_id}"):
                result = self.connector.get(
                    str(target_stage_id),
                    str(stage_id),
                    connector_get_key,
                )
        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {connector_get_key}: {e}")
            return False

        if result is None:
            return False
        payload_data, size = result

        if payload_data:
            previous_additional_information = deserialize_additional_information(
                getattr(request, "additional_information", None)
            )
            # Update connector state
            self.get_req_chunk[req_id] += 1

            meta = payload_data.get("meta", {})
            if self.model_mode == "ar":
                merged_payload = self._update_request_payload(external_req_id, payload_data)
                request.additional_information = preserve_scheduling_metadata(
                    previous_additional_information,
                    merged_payload,
                )
                if meta.get("finished"):
                    self.finished_requests.add(req_id)
            else:
                if meta.get("finished"):
                    self.finished_requests.add(req_id)

                new_ids = payload_data.get("codes", {}).get("audio", [])
                request.prompt_token_ids = new_ids
                info = dict(previous_additional_information)
                for key, value in payload_data.items():
                    if key == "codes":
                        continue
                    if isinstance(value, dict):
                        existing_sub = info.get(key)
                        merged_sub = dict(existing_sub) if isinstance(existing_sub, dict) else {}
                        for sk, sv in value.items():
                            if key == "meta" and sk == "finished":
                                continue
                            merged_sub[sk] = sv
                        info[key] = merged_sub
                        continue
                    info[key] = value
                request.additional_information = preserve_scheduling_metadata(
                    previous_additional_information,
                    info,
                )
                request.num_computed_tokens = 0

                # Empty chunk with more data expected: keep polling.
                if not new_ids and not meta.get("finished"):
                    return True

            # Mark as finished for consumption
            nvtx_mark(
                f"s{stage_id}_chunk_ready:req={str(req_id)[-8:]}:"
                f"chunk={chunk_id}:finished={int(bool(meta.get('finished')))}"
            )
            self._finished_load_reqs.add(req_id)
            logger.debug(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
            return True

        return False

    def _update_request_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Update the stored payload for *req_id* with the latest chunk."""
        if req_id not in self.request_payload:
            self.request_payload[req_id] = payload_data
            return payload_data
        origin = self.request_payload[req_id]
        raw_ok = payload_data.get("meta", {}).pop("override_keys", [])
        override_keys = {tuple(k) if isinstance(k, list) else k for k in raw_ok}

        for type_key, new_val in payload_data.items():
            if not isinstance(new_val, dict):
                continue
            origin_sub = origin.get(type_key)
            if not isinstance(origin_sub, dict):
                continue
            for qual, value in new_val.items():
                if type_key == "meta" and qual == "finished":
                    continue
                if (type_key, qual) in override_keys:
                    continue
                if isinstance(value, torch.Tensor) and qual in origin_sub:
                    new_val[qual] = torch.cat([origin_sub[qual], value], dim=0)
                elif isinstance(value, list) and qual in origin_sub:
                    new_val[qual] = origin_sub[qual] + value

        self.request_payload[req_id] = payload_data
        return payload_data

    def _send_single_request(self, task: dict):
        raw_po = task["pooling_output"]
        pooling_output = unflatten_payload(raw_po) if isinstance(raw_po, dict) else raw_po
        request = task["request"]
        is_finished = task["is_finished"]
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        external_req_id = request.external_req_id
        # Payload builders own creation of per-request chunk state when they
        # actually need it.  Reading the prospective chunk id must not leave a
        # defaultdict entry behind for a request with no downstream payload.
        chunk_id = self.put_req_chunk.get(external_req_id, 0)
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
        # Process payload in save_loop thread
        payload_data = None
        if self.custom_process_next_stage_input_func:
            try:
                with nvtx_range(f"s{stage_id}_build_payload:req={str(external_req_id)[-8:]}:chunk={chunk_id}"):
                    payload_data = self.custom_process_next_stage_input_func(
                        transfer_manager=self,
                        pooling_output=pooling_output,
                        request=request,
                        is_finished=is_finished,
                    )

            except Exception as e:
                raise RuntimeError(
                    "custom next-stage payload builder failed for "
                    f"request {external_req_id}: {e}"
                ) from e

        if not payload_data:
            # A non-terminal empty result can mean that an async builder is
            # buffering a partial chunk, so its sender state remains live.  A
            # terminal empty result is safe only when metadata proves that this
            # stage is the request's final stage and no data has ever been
            # buffered or sent downstream.  Otherwise an empty terminal result
            # would omit the downstream EOF marker, so fail drain closed rather
            # than hiding a real protocol error.
            if is_finished:
                try:
                    request_info = deserialize_additional_information(
                        getattr(request, "additional_information", None)
                    )
                except Exception as e:
                    raise RuntimeError(
                        "terminal request produced no payload and final-stage "
                        f"metadata could not be decoded for {external_req_id}: {e}"
                    ) from e

                final_stage_id = request_info.get("omni_final_stage_id")
                current_stage_id = self.connector.stage_id
                safe_local_terminal = (
                    type(final_stage_id) is int
                    and type(current_stage_id) is int
                    and final_stage_id <= current_stage_id
                    and self.put_req_chunk.get(external_req_id, 0) == 0
                    and external_req_id not in self.request_payload
                    and external_req_id not in self.code_prompt_token_ids
                )
                if not safe_local_terminal:
                    raise RuntimeError(
                        "terminal request produced no payload without a proven "
                        "local-final route or with live sender state: "
                        f"request={external_req_id}, final_stage_id={final_stage_id!r}, "
                        f"current_stage_id={current_stage_id!r}, "
                        f"put_chunks={self.put_req_chunk.get(external_req_id, 0)!r}, "
                        f"has_payload={external_req_id in self.request_payload}, "
                        f"has_codes={external_req_id in self.code_prompt_token_ids}"
                    )
                self.cleanup_sender(external_req_id)
            return

        nvtx_mark(f"s{stage_id}_send_enqueue:req={str(external_req_id)[-8:]}:chunk={chunk_id}")
        with nvtx_range(f"s{stage_id}_connector_put:req={str(external_req_id)[-8:]}:key={connector_put_key}"):
            success, size, metadata = self.connector.put(
                from_stage=str(stage_id),
                to_stage=str(next_stage_id),
                put_key=connector_put_key,
                data=payload_data,
            )

        if not success:
            raise RuntimeError(
                "connector put returned unsuccessful for "
                f"key={connector_put_key}, metadata={metadata!r}"
            )

        self.put_req_chunk[external_req_id] += 1
        logger.debug(f"[Stage-{stage_id}] Sent {connector_put_key}")
        finished_flag = payload_data.get("meta", {}).get("finished", payload_data.get("finished"))
        is_payload_finished = False
        if isinstance(finished_flag, torch.Tensor):
            is_payload_finished = finished_flag.numel() == 1 and bool(finished_flag.item())
        elif finished_flag is not None:
            is_payload_finished = bool(finished_flag)

        # A successful connector put is not sufficient for a terminal save:
        # the receiver also needs an explicit EOF marker.  Preserve the sender
        # bookkeeping and fail closed if the payload violates that protocol;
        # save_loop will persist the exception in _background_fault.
        if is_finished and not is_payload_finished:
            raise RuntimeError(
                "terminal save task emitted a payload without finished=true: "
                f"request={external_req_id}, key={connector_put_key}, "
                f"finished={finished_flag!r}"
            )

        # Reclaim per-request async state only after the terminal payload
        # has been sent successfully. This avoids cleanup->save races.
        if is_payload_finished:
            self.cleanup(request.request_id, external_req_id)

        if is_finished:
            self.code_prompt_token_ids.pop(external_req_id, None)
            cached_ic = getattr(self, "_cached_ic", None)
            if cached_ic is not None:
                cached_ic.pop(external_req_id, None)

    ########################################################################
    # Cleanup
    ########################################################################

    def cleanup_receiver(self, request_id: str) -> None:
        """Reclaim receiver-side per-request state (keyed by internal id).

        Safe to call from the scheduler even when ``save_async()`` has
        enqueued work that the background thread has not yet processed,
        because it only touches receiver-side dictionaries.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.finished_requests.discard(request_id)
        self.get_req_chunk.pop(request_id, None)
        self.requests_with_ready_chunks.discard(request_id)
        self.request_ids_mapping.pop(request_id, None)
        self.requests_origin_status.pop(request_id, None)

        self._cancelled_load_reqs.add(request_id)
        self._finished_load_reqs.discard(request_id)

    def cleanup_sender(self, external_req_id: str) -> None:
        """Reclaim sender-side per-request state (keyed by external id).

        Must only be called after the terminal chunk has actually been
        sent (i.e. from ``_send_single_request``), not before.

        Idempotent: calling with an already-cleaned or unknown id is safe.
        """
        self.put_req_chunk.pop(external_req_id, None)
        self.request_payload.pop(external_req_id, None)
        self.code_prompt_token_ids.pop(external_req_id, None)

        cached_ic = getattr(self, "_cached_ic", None)
        if cached_ic is not None:
            cached_ic.pop(external_req_id, None)

    def cleanup(
        self,
        request_id: str,
        external_req_id: str | None = None,
    ) -> None:
        """Reclaim all per-request state after a request finishes.

        Idempotent: calling with an already-cleaned or unknown id is safe.

        Args:
            request_id: Internal request id (receive / scheduler side key).
            external_req_id: External request id (send / payload side key).
                When *None*, looked up from ``request_ids_mapping``.
        """
        if external_req_id is None:
            external_req_id = self.request_ids_mapping.get(request_id, request_id)

        self.cleanup_receiver(request_id)
        self.cleanup_sender(external_req_id)

    def get_drain_status(self) -> dict[str, Any]:
        """Return transfer-thread and chunk bookkeeping without mutation."""

        # Keep the barrier held while checking pending/in-flight state and
        # reading detail.  Both workers must acquire this lock before changing
        # pending -> in-flight, so an all-zero snapshot cannot race with a
        # worker starting after the base counters were copied.
        with self._drain_state_lock:
            counters = {
                "pending_load_requests": len(self._pending_load_reqs),
                "pending_save_requests": len(self._pending_save_reqs),
                "recv_inflight": self._recv_inflight,
                "save_inflight": self._save_inflight,
            }
            background_active = bool(
                self._recv_inflight or self._save_inflight
            )
            background_fault = self._background_fault
            if not background_active:
                cached_ic = getattr(self, "_cached_ic", None)
                counters.update(
                    {
                        "finished_load_requests": len(self._finished_load_reqs),
                        "finished_save_requests": len(self._finished_save_reqs),
                        "finished_requests": len(self.finished_requests),
                        "requests_with_ready_chunks": len(
                            self.requests_with_ready_chunks
                        ),
                        "waiting_for_chunk_waiting_requests": len(
                            self.waiting_for_chunk_waiting_requests
                        ),
                        "waiting_for_chunk_running_requests": len(
                            self.waiting_for_chunk_running_requests
                        ),
                        "put_request_chunks": len(self.put_req_chunk),
                        "get_request_chunks": len(self.get_req_chunk),
                        "request_payloads": len(self.request_payload),
                        "code_prompt_token_ids": len(
                            self.code_prompt_token_ids
                        ),
                        "request_id_mappings": len(self.request_ids_mapping),
                        "request_origin_statuses": len(
                            self.requests_origin_status
                        ),
                        "cached_intermediate_contexts": (
                            len(cached_ic) if cached_ic is not None else 0
                        ),
                    }
                )

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
        stable_detail = not background_active
        status = {
            "healthy": healthy,
            "drained": healthy
            and stable_detail
            and all(value == 0 for value in counters.values()),
            "stable_detail": stable_detail,
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

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> None:
        """
        Process pending chunks for waiting and running queues.
        """
        if self.connector.stage_id == 0:
            return
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, self._finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, self._finished_load_reqs
        )
        while len(running_queue) > self.scheduler_max_num_seqs:
            request = running_queue.pop()
            request.status = RequestStatus.PREEMPTED
            waiting_queue.prepend_requests([request])

    def restore_queues(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.
        """
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            running_queue.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if request.request_id in self.finished_requests:
                    continue
                # Requests that waiting for chunk
                self.load_async(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
                    request.status = target_status
                    finished_load_reqs.remove(request.request_id)
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            self.requests_origin_status[request.request_id] = target_status
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)

    def finish_requests(
        self, request_ids: Any, finished_status: RequestStatus, requests: dict[str, Request] | None = None
    ) -> list[tuple[str, int]]:
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = requests.keys()

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = requests.get(req_id) if requests else None
            if request is None or request.is_finished():
                # Invalid request ID.
                continue
            if req_id in self.requests_origin_status:
                request.status = self.requests_origin_status.pop(req_id)
