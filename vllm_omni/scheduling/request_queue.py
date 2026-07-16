"""Dynamically ordered request queue used by non-native baseline policies."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request

from vllm_omni.scheduling.policy import (
    BaselineSchedulingPolicy,
    PolicyKey,
    policy_applies_to_stage,
    policy_key,
)


class PolicyOrderedRequestQueue(RequestQueue):
    """A small O(n log n) queue whose keys may change between scheduler ticks."""

    def __init__(self, key: Callable[[Request], PolicyKey]) -> None:
        self._requests: list[Request] = []
        self._key = key

    def _ordered(self) -> list[Request]:
        return sorted(self._requests, key=self._key)

    def add_request(self, request: Request) -> None:
        self._requests.append(request)

    def pop_request(self) -> Request:
        request = self.peek_request()
        self._requests.remove(request)
        return request

    def peek_request(self) -> Request:
        if not self._requests:
            raise IndexError("peek from an empty policy queue")
        return min(self._requests, key=self._key)

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        self._requests.extend(requests)

    def remove_request(self, request: Request) -> None:
        self._requests.remove(request)

    def remove(self, request: Request) -> None:
        """Match the deque-like interface used by chunk transfer adapters.

        vLLM's ``RequestQueue`` protocol calls this operation
        ``remove_request``, while the Omni chunk/input coordinators also use
        the waiting queue as a deque and call ``remove`` directly.
        """
        self.remove_request(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests_to_remove = set(requests)
        self._requests = [
            request
            for request in self._requests
            if request not in requests_to_remove
        ]

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._ordered())


def maybe_create_policy_ordered_queue(
    native_queue: RequestQueue,
    *,
    policy: BaselineSchedulingPolicy,
    stage_id: int,
    data_ready_time: Callable[[Request], float],
    clock: Callable[[], float] = time.monotonic,
) -> RequestQueue:
    """Return the native queue unchanged unless a custom policy applies."""

    if not policy_applies_to_stage(policy, stage_id):
        return native_queue
    if native_queue:
        raise ValueError("baseline policy queue must replace an empty native queue")
    return PolicyOrderedRequestQueue(
        key=lambda request: policy_key(
            request,
            policy=policy,
            stage_id=stage_id,
            data_ready_time=data_ready_time(request),
            now=clock(),
        )
    )
