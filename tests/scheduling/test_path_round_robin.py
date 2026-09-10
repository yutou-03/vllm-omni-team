from types import SimpleNamespace

import pytest

from vllm_omni.scheduling.path_round_robin import PathRoundRobin
from vllm_omni.scheduling.request_queue import PolicyOrderedRequestQueue


def setup_rr(text=1, audio=2, phase=0):
    rr = PathRoundRobin(text, audio, phase)
    key = lambda r: rr.key(r, path=r.path, ready_order=r.order)
    queues = [PolicyOrderedRequestQueue(key), PolicyOrderedRequestQueue(key)]
    running, events = [], []

    def choose():
        return rr.select(queues, running=running, key=key,
                         path=lambda r: r.path, emit=events.append)

    def add(name, path, order, queue=0):
        r = SimpleNamespace(request_id=name, path=path, order=order)
        queues[queue].add_request(r)
        return r

    return rr, queues, running, events, choose, add


def test_multiple_admissions_in_one_tick_and_last_reconciliation():
    rr, queues, running, events, choose, add = setup_rr()
    for i in range(6):
        add(f"t{i}", "text", i)
        add(f"a{i}", "audio", i)
    paths = []
    for _ in range(6):
        request = choose().pop_request()
        paths.append(request.path)
        running.append(request)
    rr.reconcile(running, events.append)
    assert paths == ["text", "audio", "audio"] * 2
    assert rr.admissions == 6
    assert all(e["admitted"] for e in events)


def test_skipped_and_failed_admission_do_not_consume_opportunity():
    rr, queues, running, events, choose, add = setup_rr()
    blocked = add("blocked", "text", 0)
    add("audio", "audio", 1)
    assert choose().pop_request() is blocked  # native scheduler skipped it
    request = choose().pop_request()
    assert rr.cursor == 0
    running.append(request)
    rr.reconcile(running, events.append)
    assert [e["admitted"] for e in events] == [False, True]
    assert events[-1]["borrowed"]
    assert rr.cursor == 1


def test_peek_without_pop_and_full_slot_do_not_advance():
    rr, queues, running, events, choose, add = setup_rr()
    add("t", "text", 0)
    choose().peek_request()
    rr.reconcile(running, events.append)
    rr.reconcile(running, events.append)
    assert rr.cursor == rr.admissions == 0


def test_cross_queue_fcfs_and_empty_queue_borrowing():
    rr, queues, running, events, choose, add = setup_rr()
    old = add("old", "audio", 0, queue=1)
    add("new", "audio", 1)
    assert choose().pop_request() is old
    running.append(old)
    rr.reconcile(running, events.append)
    assert events[0]["borrowed"]
    assert events[0]["visible_candidate_counts"] == {"text": 0, "audio": 2}


@pytest.mark.parametrize("raw", ["", "0:1", "1:0", "-1:2", "1.0:2", "1:2:3"])
def test_invalid_ratio_fails_closed(raw):
    with pytest.raises(ValueError):
        PathRoundRobin.from_environment({"VLLM_OMNI_PATH_RR": raw})


def test_phase_and_unset():
    assert PathRoundRobin.from_environment({}) is None
    assert PathRoundRobin(1, 2, 1).target == "audio"
    with pytest.raises(ValueError):
        PathRoundRobin(1, 2, 3)
    with pytest.raises(ValueError):
        PathRoundRobin.from_environment({"VLLM_OMNI_PATH_RR_PHASE": "0"})
