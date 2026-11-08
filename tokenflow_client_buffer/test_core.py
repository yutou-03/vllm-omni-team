import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest

from tokenflow_client_buffer.core import CapacityEstimate, CapacityEstimator, ClientFeedback, ConsumerBuffer, WorkingSet, stream_contract
from tokenflow_client_buffer.scheduler import ClientBufferScheduler


def estimate(gamma=10., at=0., until=100., modality="text"):
    return CapacityEstimate(gamma, at, until, modality, "test-profile", "test")


class BufferTests(unittest.TestCase):
    def test_receive_consume_stall_resume_and_tail(self):
        b = ConsumerBuffer(2.)
        b.receive(4, 10)
        b.tick(11)
        self.assertEqual(b.buffered, 2)
        b.tick(13)
        self.assertEqual(b.stalled_s, 1)
        b.receive(2, 14)
        self.assertEqual((b.stalled_s, b.max_stall_s, b.stall_count), (2, 2, 1))
        b.finish(14)
        b.tick(20)
        self.assertEqual((b.completed_at, b.consumed, b.stalled_s), (15, 6, 2))
        self.assertEqual(b.received, b.consumed + b.buffered)

    def test_startup_is_not_rebuffer_and_short_stream_drains(self):
        b = ConsumerBuffer(1., startup_units=3.)
        b.receive(1, 0)
        b.tick(10)
        self.assertIsNone(b.started_at)
        self.assertEqual(b.stalled_s, 0)
        b.finish(10)
        b.tick(12)
        self.assertEqual(b.completed_at, 11)

    def test_audio_chunk_partition_does_not_change_playout(self):
        a, b = ConsumerBuffer(1.), ConsumerBuffer(1.)
        a.receive(2, 0)
        for _ in range(20): b.receive(.1, 0)
        a.finish(0)
        b.finish(0)
        a.tick(3)
        b.tick(3)
        self.assertAlmostEqual(a.completed_at, b.completed_at)
        self.assertEqual(a.stalled_s, b.stalled_s)

    def test_feedback_has_no_credit_for_stage_inventory(self):
        b = ConsumerBuffer(2.)
        b.receive(10, 1)
        raw = b.feedback(1, request_id="r", session="s", modality="text", clock_domain="host")
        kw = dict(now=1., session="s", modality="text", clock_domain="host", rate=2.)
        f = ClientFeedback.parse(raw, **kw)
        self.assertEqual(f.coverage(2, 2), 4)
        self.assertIsNone(f.coverage(4, 2))
        raw["receiver_ready_inventory_units"] = 1000000
        self.assertEqual(ClientFeedback.parse(raw, **kw).coverage(2, 2), 4)
        raw["source"] = "stage_receiver"
        with self.assertRaises(ValueError): ClientFeedback.parse(raw, **kw)

    def test_feedback_order_clock_session_rate_and_conservation(self):
        b = ConsumerBuffer(1.)
        b.receive(2, 1)
        raw = b.feedback(1, request_id="r", session="s", modality="audio", clock_domain="host")
        kw = dict(now=2., session="s", modality="audio", clock_domain="host", rate=1.)
        f = ClientFeedback.parse(raw, **kw)
        for field, value in [("observed_at", 3.), ("consumed", 3.), ("session", "old"), ("rate", 2.), ("clock_domain", "remote"), ("sequence", True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                ClientFeedback.parse({**raw, field: value}, **kw)
        with self.assertRaises(ValueError): ClientFeedback.parse(raw, previous=f, **kw)

    def test_speed_must_be_inside_transmitted_payload(self):
        p = dict(stream=True, omni_scheduling=dict(schema_version=2, source_request_id="r", request_path="text", text_consume_rate_tps=7.))
        self.assertEqual(stream_contract(p, "text"), ("r", 7.))
        del p["omni_scheduling"]["text_consume_rate_tps"]
        p["text_consume_rate_tps"] = 7.
        with self.assertRaises(KeyError): stream_contract(p, "text")


class CapacityTests(unittest.TestCase):
    def test_admission_equal_capacity_and_fifo_no_skip(self):
        w = WorkingSet("text", recovery_samples=1)
        for rid, rate in [("a", 6), ("b", 5), ("c", 4)]: w.add(rid, rate)
        self.assertEqual(w.admit_next(estimate(), 1), "a")
        self.assertIsNone(w.admit_next(estimate(), 1))
        self.assertEqual(w.mode, "buffer")
        w.complete("a")
        self.assertEqual(w.admit_next(estimate(), 2), "b")
        self.assertEqual(w.admit_next(estimate(), 2), "c")
        self.assertEqual(w.demand, 9)
        w.add("d", 1)
        self.assertEqual(w.admit_next(estimate(), 2), "d")
        self.assertEqual(w.demand, 10)

    def test_capacity_drop_preserves_working_set_and_immediate_fallback(self):
        w = WorkingSet("audio", recovery_samples=1)
        w.add("a", 1)
        w.add("b", 1)
        w.admit_next(estimate(2, modality="audio"), 1)
        w.admit_next(estimate(2, modality="audio"), 1)
        self.assertEqual(w.update(estimate(1, at=2, modality="audio"), 2), "fcfs")
        self.assertEqual((w.demand, len(w.members), w.reason), (2, 2, "overload"))

    def test_stale_capacity_and_hysteresis(self):
        w = WorkingSet("text", recovery_samples=2)
        e = estimate(until=2)
        self.assertEqual(w.update(e, 1), "fcfs")
        self.assertEqual(w.update(e, 1.1), "fcfs")
        self.assertEqual(w.update(estimate(at=1), 1.2), "buffer")
        self.assertEqual(w.update(e, 3), "fcfs")

    def test_single_unsatisfiable_request_runs_fcfs_not_deadlock(self):
        w = WorkingSet("text", recovery_samples=1)
        w.add("a", 11)
        self.assertEqual(w.admit_next(estimate(), 1), "a")
        self.assertEqual(w.mode, "fcfs")

    def test_atomic_admissions_do_not_overcommit(self):
        w = WorkingSet("text", recovery_samples=1)
        for i in range(30): w.add(str(i), 3)
        threads = [threading.Thread(target=w.admit_next, args=(estimate(), 1)) for _ in range(30)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(w.demand, 9)

    def test_overload_freezes_new_admissions_until_existing_demand_fits(self):
        w = WorkingSet("text", recovery_samples=1)
        high = CapacityEstimate(10., 0., 100., "text", "p", "test")
        low = CapacityEstimate(2., 1., 100., "text", "p", "test")
        w.add("a", 5.)
        self.assertEqual(w.admit_next(high, 1), "a")
        w.add("b", 1.)
        self.assertIsNone(w.admit_next(low, 2))
        self.assertEqual(w.mode, "fcfs")
        self.assertEqual(w.demand, 5.)
        w.complete("a")
        self.assertEqual(w.admit_next(low, 3), "b")

    def test_idle_observation_cannot_lower_or_refresh_capacity(self):
        e = CapacityEstimator("text", safety=.9, ttl_s=10)
        old = e.seed(100, 0, "p")
        self.assertIs(e.observe(generated_units=1, duration_s=2, now=5, saturated=False, profile_id="p"), old)
        self.assertFalse(old.valid(11, "text"))
        low = e.observe(generated_units=100, duration_s=2, now=6, saturated=True, profile_id="p")
        self.assertEqual(low.gamma, 45)


def request(rid, order, rate=1.):
    return NS(request_id=rid, arrival_time=float(order), status=NS(name="RUNNING"), num_computed_tokens=10, num_prompt_tokens=3, num_output_tokens=8, num_output_placeholders=0, spec_token_ids=[], num_tokens=11,
              client_buffer_test_metadata=dict(sched_source_request_id=rid, sched_ingress_order=order, sched_text_consume_rate_tps=rate))


class MetricTests(unittest.TestCase):
    def test_text_metric_uses_discrete_token_weights(self):
        from .analyze import arrival_utility
        self.assertAlmostEqual(arrival_utility(0, 20, 100, "text"), 14.5)
        self.assertAlmostEqual(arrival_utility(0, 12, 100, "text") +
            arrival_utility(12, 8, 100, "text"), 14.5)

    def test_audio_metric_is_chunk_partition_invariant(self):
        from .analyze import arrival_utility
        self.assertAlmostEqual(arrival_utility(0, 2, 10, "audio"),
            arrival_utility(0, .7, 10, "audio") + arrival_utility(.7, 1.3, 10, "audio"))


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "snapshot.json"
        self.now = 10.
        self.events = []
        self.a = request("a", 0)
        self.s = NS(running=[self.a], waiting=[request("new", 1)], requests={"a": self.a}, max_num_running_reqs=4, kv_cache_manager=NS(), _omni_stage_id_for_trace=lambda:0)
        self.c = ClientBufferScheduler(self.s, dict(execution_cap=1, snapshot_path=str(self.path), resume_to_client_s=.5, session="s", modality="text"), lambda event, **fields:self.events.append((event, fields)), clock=lambda:self.now)
        self.write()

    def write(self, **updates):
        d = dict(schema_version=1, session="s", modality="text", written_at=self.now, mode="buffer", demand=1., gamma=10., capacity_valid_until=100., members={"a": dict(rate=1)}, feedback={})
        d.update(updates)
        self.path.write_text(json.dumps(d))

    def test_no_client_feedback_no_pause_despite_stage_stock(self):
        self.write(receiver_ready_inventory_units=1000000)
        with self.c.scheduling_context(): self.assertEqual(self.s.running, [self.a])
        self.assertFalse(self.c.held_since)

    def test_client_credit_pause_restore_identity_then_capacity_fallback(self):
        fb = dict(request_id="a", session="s", modality="text", rate=1., received=10., consumed=0., observed_at=10., sequence=1, started=True, complete=False)
        self.write(feedback={"a": fb})
        with self.c.scheduling_context():
            self.assertEqual(self.s.running, [])
            self.assertEqual(self.s.max_num_running_reqs, 1)
        self.assertEqual(self.s.running, [self.a])
        self.assertIn("a", self.c.held_since)
        self.now = 10.1
        self.write(gamma=.5)
        with self.c.scheduling_context(): self.assertEqual(self.s.running, [self.a])
        self.assertFalse(self.c.held_since)
        self.assertEqual(self.c.last_mode, "fcfs")
        self.assertEqual(self.s.max_num_running_reqs, 4)

    def test_resume_before_deadline_swaps_only_buffered_owner(self):
        b = request("b", 1)
        self.s.running.append(b)
        self.s.requests["b"] = b
        self.c.active_ids = {"b"}
        self.c.held_since["a"] = 9.
        self.c.resume_deadlines["a"] = 10.08
        feedback = {rid: dict(request_id=rid, session="s", modality="text", rate=1.,
            received=units, consumed=0., observed_at=10., sequence=1,
            started=True, complete=False) for rid, units in (("a", .82), ("b", 10.))}
        self.write(demand=2., members={"a": dict(rate=1), "b": dict(rate=1)}, feedback=feedback)
        with self.c.scheduling_context():
            self.assertEqual(self.s.running, [self.a])
        self.c.record_scheduled(NS(num_scheduled_tokens={"a": 1}))
        self.assertNotIn("a", self.c.held_since)
        self.assertIn("b", self.c.held_since)
        self.assertIsNone(self.c.latched_fault)

    def test_new_client_receipts_extend_old_pause_deadline(self):
        b = request("b", 1)
        self.s.running.append(b)
        self.s.requests["b"] = b
        self.c.active_ids = {"b"}
        self.c.held_since["a"] = 9.
        self.c.resume_deadlines["a"] = 9.9
        self.s.waiting = []
        fb = dict(request_id="a", session="s", modality="text", rate=1., received=10.,
            consumed=0., observed_at=10., sequence=2, started=True, complete=False)
        self.write(demand=2., members={"a":dict(rate=1),"b":dict(rate=1)}, feedback={"a":fb})
        with self.c.scheduling_context():self.assertEqual(self.s.running,[b])
        self.assertIsNone(self.c.latched_fault)
        self.assertEqual(self.c.last_mode,"buffer")
        self.assertGreater(self.c.resume_deadlines["a"], self.now)

    def test_unbuffered_slot_owner_is_not_replaced_by_waiting_resident(self):
        b = request("b", 1)
        self.s.running.append(b)
        self.s.requests["b"] = b
        self.c.active_ids = {"b"}
        self.write(demand=2., members={"a": dict(rate=1), "b": dict(rate=1)})
        with self.c.scheduling_context():
            self.assertEqual(self.s.running, [b])
        self.assertFalse(self.c.held_since)

    def test_native_resident_waiter_gets_slot_without_being_marked_paused(self):
        b = request("b", 1)
        self.s.running.append(b)
        self.s.requests["b"] = b
        self.s.waiting=[]
        self.c.active_ids={"a"}
        def fb(rid, units):
            return dict(request_id=rid, session="s", modality="text", rate=1., received=units,
                consumed=0., observed_at=10., sequence=1, started=True, complete=False)
        self.write(demand=2.,members={"a":dict(rate=1),"b":dict(rate=1)},feedback={"a":fb("a",10.),"b":fb("b",0.)})
        with self.c.scheduling_context():self.assertEqual(self.s.running,[b])
        self.assertEqual(set(self.c.held_since),{"a"})
        self.assertTrue(all(fields["coverage_s"]>0 for event,fields in self.events if event=="client_buffer_pause"))

    def test_unselected_native_resident_with_empty_buffer_is_not_a_policy_pause(self):
        b=request("b",1);self.s.running.append(b);self.s.requests["b"]=b;self.c.active_ids={"a"}
        fb=dict(request_id="b",session="s",modality="text",rate=1.,received=0.,consumed=0.,
                observed_at=10.,sequence=1,started=True,complete=False)
        self.write(demand=2.,members={"a":dict(rate=1),"b":dict(rate=1)},feedback={"b":fb})
        with self.c.scheduling_context():self.assertEqual(self.s.running,[self.a])
        self.assertFalse(self.c.held_since)

    def test_cancelled_hidden_request_is_not_resurrected(self):
        fb = dict(request_id="a", session="s", modality="text", rate=1., received=10., consumed=0., observed_at=10., sequence=1, started=True, complete=False)
        self.write(feedback={"a": fb})
        with self.c.scheduling_context(): del self.s.requests["a"]
        self.assertEqual(self.s.running, [])

    def test_exception_restores_queue_and_cap(self):
        with self.assertRaises(RuntimeError):
            with self.c.scheduling_context(): raise RuntimeError("model schedule failed")
        self.assertEqual(self.s.running, [self.a])
        self.assertEqual(self.s.max_num_running_reqs, 4)


if __name__ == "__main__": unittest.main()
