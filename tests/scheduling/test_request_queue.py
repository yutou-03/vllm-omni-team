from __future__ import annotations

from vllm.v1.core.sched.request_queue import FCFSRequestQueue

from vllm_omni.scheduling.policy import BaselineSchedulingPolicy
from vllm_omni.scheduling.request_queue import (
    PolicyOrderedRequestQueue,
    maybe_create_policy_ordered_queue,
)


class FakeRequest:
    def __init__(
        self,
        request_id: str,
        score: float,
        scheduling: dict | None = None,
    ) -> None:
        self.request_id = request_id
        self.score = score
        self.additional_information = (
            {"meta": scheduling} if scheduling is not None else None
        )


def test_policy_queue_recomputes_dynamic_keys_on_each_access():
    first = FakeRequest("first", 10)
    second = FakeRequest("second", 20)
    queue = PolicyOrderedRequestQueue(key=lambda request: (request.score, request.request_id))
    queue.add_request(first)
    queue.add_request(second)

    assert queue.peek_request() is first
    second.score = 5
    assert queue.peek_request() is second
    assert list(queue) == [second, first]
    assert queue.pop_request() is second
    assert queue.pop_request() is first


def test_policy_queue_supports_remove_and_prepend_interface():
    first = FakeRequest("first", 30)
    second = FakeRequest("second", 10)
    third = FakeRequest("third", 20)
    queue = PolicyOrderedRequestQueue(key=lambda request: (request.score, request.request_id))

    queue.prepend_request(first)
    other = PolicyOrderedRequestQueue(key=lambda request: (request.score, request.request_id))
    other.add_request(second)
    other.add_request(third)
    queue.prepend_requests(other)
    assert list(queue) == [second, third, first]

    queue.remove_request(third)
    assert list(queue) == [second, first]
    queue.remove(first)
    assert list(queue) == [second]
    queue.remove_requests([second])
    assert list(queue) == []


def test_native_fcfs_factory_returns_exact_original_queue():
    native_queue = FCFSRequestQueue()
    wrapped = maybe_create_policy_ordered_queue(
        native_queue,
        policy=BaselineSchedulingPolicy.NATIVE_FCFS,
        stage_id=0,
        data_ready_time=lambda request: 0.0,
    )

    assert wrapped is native_queue


def test_stage_deadline_edf_wraps_every_stage_queue():
    for stage_id in (0, 1, 2):
        native_queue = FCFSRequestQueue()
        wrapped = maybe_create_policy_ordered_queue(
            native_queue,
            policy=BaselineSchedulingPolicy.STAGE_DEADLINE_EDF_NP,
            stage_id=stage_id,
            data_ready_time=lambda request: 0.0,
        )
        assert isinstance(wrapped, PolicyOrderedRequestQueue)


def test_custom_policy_factory_orders_requests_by_shared_policy_key():
    native_queue = FCFSRequestQueue()
    wrapped = maybe_create_policy_ordered_queue(
        native_queue,
        policy=BaselineSchedulingPolicy.FINAL_DEADLINE_EDF_NP,
        stage_id=0,
        data_ready_time=lambda request: request.score,
        clock=lambda: 100.0,
    )
    later = FakeRequest(
        "later",
        1.0,
        {
            "sched_source_request_id": "later",
            "sched_ingress_order": 0,
            "sched_deadline_monotonic_s": 20.0,
        },
    )
    earlier = FakeRequest(
        "earlier",
        2.0,
        {
            "sched_source_request_id": "earlier",
            "sched_ingress_order": 1,
            "sched_deadline_monotonic_s": 10.0,
        },
    )

    wrapped.add_request(later)
    wrapped.add_request(earlier)

    assert isinstance(wrapped, PolicyOrderedRequestQueue)
    assert wrapped.pop_request() is earlier
