import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest

from .pause import PausedRequest

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

    def test_first_short_audio_chunk_starts_immediately_and_drains(self):
        b = ConsumerBuffer(1.)
        b.tick(0)
        b.tick(10)
        self.assertIsNone(b.started_at)
        self.assertEqual(b.stalled_s, 0)
        b.receive(.05, 10)  # 50 ms, below the removed 200 ms startup threshold.
        self.assertEqual(b.first_ready, 10)
        self.assertEqual(b.started_at, b.first_ready)
        b.finish(10)
        b.tick(12)
        self.assertEqual(b.completed_at, 10.05)
        self.assertEqual(b.stalled_s, 0)

    def test_gap_after_first_audio_chunk_counts_as_stall(self):
        b = ConsumerBuffer(1.)
        b.receive(.05, 10)
        b.tick(10.2)
        self.assertAlmostEqual(b.consumed, .05)
        self.assertAlmostEqual(b.stalled_s, .15)
        b.receive(.05, 10.2)
        self.assertEqual(b.started_at, 10)
        b.finish(10.2)
        b.tick(11)
        self.assertAlmostEqual(b.completed_at, 10.25)
        self.assertAlmostEqual(b.stalled_s, .15)

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

    def test_reduced_demand_recovers_from_overload_without_new_capacity_sample(self):
        w=WorkingSet('audio',recovery_samples=1)
        e=estimate(2.,modality='audio')
        for rid in ('a','b','waiting'):w.add(rid,1.)
        w.admit_next(e,1.);w.admit_next(e,1.)
        w.pause('a');w.admit_next(e,1.)
        w.resume('a');self.assertEqual(w.update(e,1.),'fcfs')
        w.complete('b')
        self.assertEqual(w.update(e,2.),'buffer')
        self.assertEqual(w.reason,'capacity_feasible')
        self.assertEqual(w.demand,2.)

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

    def test_admission_reserve_is_not_the_physical_overload_line(self):
        w=WorkingSet('text',recovery_samples=1)
        e=CapacityEstimate(100.,0.,100.,'text','p','test',85.)
        for i in range(10):w.add(str(i),10.)
        for i in range(8):self.assertEqual(w.admit_next(e,1.),str(i))
        self.assertIsNone(w.admit_next(e,1.))
        dip=CapacityEstimate(90.,2.,100.,'text','p','test',76.5)
        self.assertEqual(w.update(dip,2.),'buffer')
        self.assertIsNone(w.admit_next(dip,2.))
        self.assertEqual(w.demand,80.)
        loss=CapacityEstimate(79.,3.,100.,'text','p','test',67.15)
        self.assertEqual(w.update(loss,3.),'fcfs')
        self.assertIsNone(w.admit_next(loss,3.))

    def test_sustained_loss_lowers_smoothed_physical_capacity(self):
        e=CapacityEstimator('text',safety=.85,alpha=.2)
        e.seed(100.,0.,'p')
        for i in range(1,7):
            out=e.observe(generated_units=80.,duration_s=2.,now=2.*i,
                          saturated=True,profile_id='p')
        self.assertLess(out.gamma,60.)
        self.assertAlmostEqual(out.admission_bound,.85*out.gamma)

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

    def test_expiry_refills_free_native_slots_without_draining_whole_batch(self):
        w = WorkingSet("audio", recovery_samples=1, native_execution_limit=32)
        for i in range(200): w.add(str(i), 1.)
        e = estimate(19.34, until=2., modality="audio")
        for _ in range(19): self.assertIsNotNone(w.admit_next(e, 1.))
        self.assertIsNone(w.admit_next(e, 1.))
        for i in range(19, 32): self.assertEqual(w.admit_next(e, 3.), str(i))
        self.assertIsNone(w.admit_next(e, 3.))
        w.complete("0")
        self.assertEqual(w.admit_next(e, 4.), "32")
        self.assertEqual(len(w.members), 32)
        low = estimate(15., at=5., until=20., modality="audio")
        self.assertIsNone(w.admit_next(low, 6.))
        self.assertEqual(w.reason, "overload")

    def test_unknown_capacity_waits_for_paused_owner_recovery(self):
        w = WorkingSet("audio", recovery_samples=1, native_execution_limit=32)
        e = estimate(100., until=2., modality="audio")
        for rid in ("active", "paused", "new"): w.add(rid, 1.)
        self.assertEqual(w.admit_next(e, 1.), "active")
        self.assertEqual(w.admit_next(e, 1.), "paused")
        w.pause("paused")
        self.assertIsNone(w.admit_next(e, 3.))
        w.resume("paused")
        self.assertEqual(w.admit_next(e, 3.), "new")

    def test_buffer_working_set_remains_rate_limited_above_native_batch_size(self):
        w = WorkingSet("text", max_inflight=256, recovery_samples=1, native_execution_limit=32)
        e = estimate(1000.)
        for i in range(100):
            w.add(str(i), 10.)
            self.assertEqual(w.admit_next(e, 1.), str(i))
        self.assertEqual(len(w.members), 100)
        self.assertEqual(w.mode, "buffer")

    def test_empty_output_cannot_create_zero_capacity_or_refresh_old_bound(self):
        e = CapacityEstimator("audio", ttl_s=10.)
        old = e.seed(20., 0., "p")
        self.assertIs(e.observe(generated_units=0., duration_s=8., now=8.,
            saturated=True, profile_id="p"), old)
        self.assertFalse(old.valid(11., "audio"))

    def test_observed_lower_bound_can_raise_capacity_without_full_batch(self):
        e = CapacityEstimator("text", safety=.9, ttl_s=10.)
        old = e.seed(100., 0., "p")
        new = e.observe(generated_units=240., duration_s=2., now=12.,
            saturated=False, profile_id="p")
        self.assertEqual(new.gamma, 120.)  # expired calibration is not EWMA input
        self.assertTrue(new.valid(13., "text"))
        self.assertEqual(new.source, "observed_output_lower_bound")

    def test_fresh_output_replaces_expired_higher_seed_without_saturation(self):
        e = CapacityEstimator("audio", ttl_s=900., min_window_s=8.)
        old = e.seed(29.26, 0., "audio")
        for t in range(8, 905, 8):
            e.observe(generated_units=200., duration_s=8., now=float(t),
                      saturated=False, profile_id="audio")
        self.assertFalse(old.valid(904., "audio"))
        self.assertTrue(e.estimate.valid(904., "audio"))
        self.assertEqual(e.estimate.gamma, 25.)
        self.assertEqual(e.estimate.source, "observed_output_lower_bound")

    def test_idle_observation_cannot_lower_or_refresh_capacity(self):
        e = CapacityEstimator("text", safety=.9, ttl_s=10)
        old = e.seed(100, 0, "p")
        self.assertIs(e.observe(generated_units=1, duration_s=2, now=5, saturated=False, profile_id="p"), old)
        self.assertFalse(old.valid(11, "text"))
        low = e.observe(generated_units=100, duration_s=2, now=6, saturated=True, profile_id="p")
        self.assertEqual(low.gamma, 90)  # physical EWMA; admission bound is 81
        self.assertEqual(low.admission_bound, 81)


def request(rid, order, rate=1.):
    return NS(request_id=rid, arrival_time=float(order), status=NS(name="RUNNING"), num_computed_tokens=10, num_prompt_tokens=3, num_output_tokens=8, num_output_placeholders=0, spec_token_ids=[], num_tokens=11,
              client_buffer_test_metadata=dict(sched_source_request_id=rid, sched_ingress_order=order, sched_text_consume_rate_tps=rate))


class MetricTests(unittest.TestCase):
    def test_effective_weight_thresholds(self):
        from .analyze import token_weight
        self.assertEqual(token_weight(10, 10, 20), 1.)
        self.assertEqual(token_weight(15, 10, 20), .5)
        self.assertEqual(token_weight(20, 10, 20), 0.)
        self.assertEqual(token_weight(30, 10, 20), 0.)

    def test_raw_and_effective_throughput_share_each_window(self):
        from .analyze import throughput_windows
        x = throughput_windows(100, 25, 10, 50)
        self.assertEqual(x["effective_unit_fraction"], .25)
        self.assertEqual(x["throughput_by_window"]["generation"]["raw_units_per_s"], 10)
        self.assertEqual(x["throughput_by_window"]["generation"]["effective_units_per_s"], 2.5)
        self.assertEqual(x["throughput_by_window"]["client_drain"]["raw_units_per_s"], 2)
        self.assertEqual(x["throughput_by_window"]["client_drain"]["effective_units_per_s"], .5)
        with self.assertRaises(ValueError):
            throughput_windows(100, 101, 10, 50)
        with self.assertRaises(ValueError):
            throughput_windows(100, 25, 0, 50)

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

    def use_native_limit(self, limit=1):
        self.s.max_num_running_reqs = limit
        self.c = ClientBufferScheduler(self.s, dict(snapshot_path=str(self.path),
            resume_to_client_s=.5, session="s", modality="text"),
            lambda event, **fields:self.events.append((event, fields)), clock=lambda:self.now)

    def test_contract_cache_invalidates_on_payload_or_request_replacement(self):
        import sys
        from unittest.mock import patch
        raw=object();r=NS(request_id='cached',additional_information=raw)
        calls=[]
        def decode(request):
            calls.append(request);return {'sched_source_request_id':str(len(calls))}
        module=NS(request_scheduling_metadata=decode)
        with patch.dict(sys.modules,{'vllm_omni.scheduling.policy':module}):
            self.assertEqual(self.c.metadata(r),self.c.metadata(r))
            self.assertEqual(len(calls),1)
            r.additional_information=object();self.c.metadata(r)
            self.assertEqual(len(calls),2)
            self.c.metadata(NS(request_id='cached',additional_information=r.additional_information))
            self.assertEqual(len(calls),3)

    def test_native_limit_uses_max_num_seqs_without_separate_cap(self):
        self.use_native_limit(64)
        with self.c.scheduling_context():
            self.assertEqual(self.s.max_num_running_reqs,64)
            self.assertEqual(self.s.running,[self.a])
        self.assertEqual(self.c.cap,64)
        self.assertTrue(self.c.native_limit)

    def test_native_limit_can_replace_full_batch_without_artificial_resident_cap(self):
        self.use_native_limit()
        fb=dict(request_id="a",session="s",modality="text",rate=1.,received=10.,
                consumed=0.,observed_at=10.,sequence=1,started=True,complete=False)
        self.write(feedback={"a":fb})
        new=self.s.waiting[0]
        with self.c.scheduling_context():
            self.assertEqual(self.s.running,[])
            self.assertEqual(self.s.max_num_running_reqs,1)
            # Simulate native allocation accepting a new request using free KV.
            self.s.waiting.remove(new)
            self.s.requests[new.request_id]=new
            self.s.running.append(new)
        self.c.record_scheduled(NS(num_scheduled_tokens={new.request_id:1}))
        self.assertEqual(self.s.running,[self.a,new])
        self.assertEqual(self.c.active_ids,{new.request_id})
        self.assertIn("a",self.c.pauses)
        self.assertEqual(self.s.max_num_running_reqs,1)

    def test_safe_buffer_can_yield_to_new_work_below_native_limit(self):
        self.use_native_limit(32)
        fb=dict(request_id="a",session="s",modality="text",rate=1.,received=10.,
                consumed=0.,observed_at=10.,sequence=1,started=True,complete=False)
        self.write(feedback={"a":fb})
        new=self.s.waiting[0]
        with self.c.scheduling_context():
            self.assertEqual(self.s.running, [])
            self.assertEqual(self.s.max_num_running_reqs,32)
            self.s.waiting.remove(new)
            self.s.requests[new.request_id]=new
            self.s.running.append(new)
        self.c.record_scheduled(NS(num_scheduled_tokens={new.request_id:1}))
        self.assertIn("a",self.c.pauses)
        event=next(e for k,e in self.events if k=="client_buffer_pause")
        self.assertEqual(event["trigger"],"waiting_work")
        self.assertLess(event["selected_before_pause"],32)
        self.assertGreater(event["coverage_s"],event["safety_threshold_s"])

    def test_below_limit_pause_waits_for_beneficiary_or_safety_deadline(self):
        self.use_native_limit(32)
        fb=dict(request_id='a',session='s',modality='text',rate=1.,received=10.,
                consumed=0.,observed_at=10.,sequence=1,started=True,complete=False)
        self.write(feedback={'a':fb})
        new=self.s.waiting[0]
        with self.c.scheduling_context():
            self.s.waiting.remove(new);self.s.requests[new.request_id]=new
            self.s.running.append(new)
        self.c.record_scheduled(NS(num_scheduled_tokens={new.request_id:1}))
        self.now=10.1;fb.update(observed_at=10.1,consumed=.1,sequence=2)
        self.write(demand=2.,members={'a':dict(rate=1),'new':dict(rate=1)},feedback={'a':fb})
        with self.c.scheduling_context():
            # Empty sequence slots do not terminate a useful safe pause after
            # one iteration while the actual beneficiary is still starting.
            self.assertEqual(self.s.running,[new])
        self.now=10.2;fb.update(observed_at=10.2,consumed=.2,sequence=3)
        newfb=dict(fb,request_id='new',received=1.,consumed=0.)
        self.write(demand=2.,members={'a':dict(rate=1),'new':dict(rate=1)},
                   feedback={'a':fb,'new':newfb})
        with self.c.scheduling_context():
            self.assertEqual({r.request_id for r in self.s.running},{'a','new'})
        self.c.record_scheduled(NS(num_scheduled_tokens={'a':1,'new':1}))
        self.assertFalse(self.c.pauses)

    def test_due_held_owner_already_selected_is_not_added_twice(self):
        self.use_native_limit(32)
        self.s.waiting=[];self.c.active_ids={'a'}
        self.c.pauses['a'] = PausedRequest(9., float("inf"), .8);self.c.pauses['a'].resume_by=10.
        with self.c.scheduling_context():
            self.assertEqual(self.s.running,[self.a])

    def test_executing_startup_can_benefit_without_a_native_waiting_queue(self):
        self.use_native_limit(32)
        b=request('b',1);self.s.running.append(b);self.s.requests['b']=b
        self.s.waiting=[];self.c.last_scheduled_ids={'a','b'}
        fb=dict(request_id='a',session='s',modality='text',rate=1.,received=3.5,
                consumed=0.,observed_at=10.,sequence=1,started=True,complete=False)
        self.write(demand=2.,members={'a':dict(rate=1),'b':dict(rate=1)},feedback={'a':fb})
        with self.c.scheduling_context():self.assertEqual(self.s.running,[b])
        self.c.record_scheduled(NS(num_scheduled_tokens={'b':1}))
        self.assertEqual(self.c.pauses['a'].beneficiary,'b')
        # The beneficiary still has no first output; safety overrides it.
        self.now=11.2;fb.update(observed_at=11.2,consumed=1.2,sequence=2)
        self.write(demand=2.,members={'a':dict(rate=1),'b':dict(rate=1)},feedback={'a':fb})
        with self.c.scheduling_context():
            self.assertEqual({r.request_id for r in self.s.running},{'a','b'})
        self.c.record_scheduled(NS(num_scheduled_tokens={'a':1,'b':1}))
        self.assertFalse(self.c.pauses)

    def test_startup_priority_survives_empty_or_alternating_native_iterations(self):
        self.use_native_limit(32)
        b=request('b',1);self.s.running.append(b);self.s.requests['b']=b
        self.s.waiting=[];self.c.last_scheduled_ids=set()
        fb=dict(request_id='a',session='s',modality='text',rate=1.,received=10.,
                consumed=0.,observed_at=10.,sequence=1,started=True,complete=False)
        self.write(demand=2.,members={'a':dict(rate=1),'b':dict(rate=1)},feedback={'a':fb})
        # b has real compute progress and is runnable now, even if the previous
        # native iteration was empty or executed a different input-ready subset.
        with self.c.scheduling_context():self.assertEqual(self.s.running,[b])
        self.assertIn('a',self.c.pauses)

    def test_opportunistic_pauses_respect_existing_minimum_slice(self):
        self.use_native_limit(2)
        b=request('b',1);self.s.running.append(b);self.s.requests['b']=b
        def fb(rid):return dict(request_id=rid,session='s',modality='text',rate=1.,received=10.,
                               consumed=0.,observed_at=self.now,sequence=1,started=True,complete=False)
        self.write(demand=2.,members={'a':dict(rate=1),'b':dict(rate=1)},feedback={'a':fb('a'),'b':fb('b')})
        new=self.s.waiting[0]
        with self.c.scheduling_context():
            self.s.waiting.remove(new);self.s.requests[new.request_id]=new;self.s.running.append(new)
        self.c.record_scheduled(NS(num_scheduled_tokens={'b':1,'new':1}))
        self.now=10.01
        self.write(demand=3.,members={x:dict(rate=1) for x in ('a','b','new')},feedback={'a':fb('a'),'b':fb('b')})
        with self.c.scheduling_context():self.assertEqual({r.request_id for r in self.s.running},{'b','new'})
        self.assertEqual(set(self.c.pauses),{'a'})
        self.now=10.06
        # A second donor must have a distinct beneficiary, not duplicate the
        # still-active pause for 'new'. The minimum-slice test uses another arrival.
        newer=request('newer',3);self.s.waiting.append(newer);self.s.requests['newer']=newer
        self.write(demand=4.,members={x:dict(rate=1) for x in ('a','b','new','newer')},feedback={'a':fb('a'),'b':fb('b')})
        with self.c.scheduling_context():self.assertEqual([r.request_id for r in self.s.running],['new'])
        self.assertIn('b',self.c.pauses)

    def test_no_pause_without_real_waiting_work_even_with_large_buffer(self):
        self.use_native_limit(32)
        self.s.waiting=[]
        fb=dict(request_id="a",session="s",modality="text",rate=1.,received=10.,
                consumed=0.,observed_at=10.,sequence=1,started=True,complete=False)
        self.write(feedback={"a":fb})
        with self.c.scheduling_context():
            self.assertEqual(self.s.running,[self.a])
        self.assertFalse(self.c.pauses)

    def test_native_limit_fcfs_preserves_native_order_and_admission(self):
        self.use_native_limit(2)
        b=request("b",1);self.s.running=[b,self.a];self.s.requests["b"]=b
        self.write(mode="fcfs",reason="baseline")
        with self.c.scheduling_context():
            self.assertEqual(self.s.running,[b,self.a])
            self.assertEqual(self.s.max_num_running_reqs,2)
        self.assertEqual(self.s.running,[b,self.a])
        self.assertFalse(self.c.pauses)

    def test_fallback_does_not_evict_unbuffered_borrower_or_erase_pause(self):
        self.use_native_limit(1)
        b=request("b",1)
        self.s.running=[self.a,b];self.s.requests["b"]=b
        self.c.active_ids={"b"}
        self.c.pauses["a"] = PausedRequest(9., float("inf"), .8)
        self.c.pauses["a"].resume_by=15.
        self.write(mode="fcfs",reason="overload",demand=2.,gamma=.5,
                   members={"a":dict(rate=1),"b":dict(rate=1)})
        with self.c.scheduling_context():
            self.assertEqual(self.s.running,[b])
        self.assertIn("a",self.c.pauses)
        self.assertFalse(any(k=="client_buffer_fallback_release" for k,_ in self.events))
        self.s.running.remove(b);del self.s.requests["b"]
        self.now=10.1;self.write(mode="fcfs",reason="overload")
        with self.c.scheduling_context():
            self.assertEqual(self.s.running,[self.a])
        self.c.record_scheduled(NS(num_scheduled_tokens={"a":1}))
        self.assertNotIn("a",self.c.pauses)

    def test_fallback_with_free_slot_keeps_pause_until_actual_execution(self):
        self.use_native_limit(32)
        self.c.active_ids=set()
        self.c.pauses['a'] = PausedRequest(9., float("inf"), .8)
        self.c.pauses['a'].resume_by=15.
        self.write(mode='fcfs',reason='overload')
        with self.c.scheduling_context():
            self.assertEqual(self.s.running,[self.a])
        self.assertIn('a',self.c.pauses)
        self.c.record_scheduled(NS(num_scheduled_tokens={'a':1}))
        self.assertNotIn('a',self.c.pauses)

    def test_fcfs_observed_recovery_can_raise_current_service_capacity(self):
        e=CapacityEstimator('text',safety=1.,alpha=1.)
        e.seed(100.,0.,'p')
        slow=e.observe(generated_units=120.,duration_s=2.,now=2.,
                       saturated=True,profile_id='p',policy='buffer')
        self.assertEqual(slow.gamma,60.)
        fast=e.observe(generated_units=200.,duration_s=2.,now=4.,
                       saturated=True,profile_id='p',policy='fcfs')
        self.assertEqual(fast.gamma,100.)
        self.assertEqual(fast.observed_at,4.)
        low=e.observe(generated_units=80.,duration_s=2.,now=6.,
                      saturated=True,profile_id='p',policy='fcfs')
        self.assertEqual(low.gamma,40.)

    def test_no_client_feedback_no_pause_despite_stage_stock(self):
        self.write(receiver_ready_inventory_units=1000000)
        with self.c.scheduling_context(): self.assertEqual(self.s.running, [self.a])
        self.assertFalse(self.c.pauses)

    def test_client_credit_pause_restore_identity_then_capacity_fallback(self):
        fb = dict(request_id="a", session="s", modality="text", rate=1., received=10., consumed=0., observed_at=10., sequence=1, started=True, complete=False)
        self.write(feedback={"a": fb})
        with self.c.scheduling_context():
            self.assertEqual(self.s.running, [])
            self.assertEqual(self.s.max_num_running_reqs, 1)
        self.assertEqual(self.s.running, [self.a])
        self.assertIn("a", self.c.pauses)
        self.now = 10.1
        self.write(gamma=.5)
        with self.c.scheduling_context(): self.assertEqual(self.s.running, [self.a])
        self.assertIn('a',self.c.pauses)
        self.c.record_scheduled(NS(num_scheduled_tokens={'a':1}))
        self.assertFalse(self.c.pauses)
        self.assertEqual(self.c.last_mode, "fcfs")
        self.assertEqual(self.s.max_num_running_reqs, 4)

    def test_resume_before_deadline_swaps_only_buffered_owner(self):
        b = request("b", 1)
        self.s.running.append(b)
        self.s.requests["b"] = b
        self.c.active_ids = {"b"}
        self.c.pauses["a"] = PausedRequest(9., float("inf"), .8)
        self.c.pauses["a"].resume_by = 10.08
        feedback = {rid: dict(request_id=rid, session="s", modality="text", rate=1.,
            received=units, consumed=0., observed_at=10., sequence=1,
            started=True, complete=False) for rid, units in (("a", .82), ("b", 10.))}
        self.write(demand=2., members={"a": dict(rate=1), "b": dict(rate=1)}, feedback=feedback)
        with self.c.scheduling_context():
            self.assertEqual(self.s.running, [self.a])
        self.c.record_scheduled(NS(num_scheduled_tokens={"a": 1}))
        self.assertNotIn("a", self.c.pauses)
        self.assertIn("b", self.c.pauses)
        self.assertIsNone(self.c.latched_fault)

    def test_new_client_receipts_extend_old_pause_deadline(self):
        b = request("b", 1)
        self.s.running.append(b)
        self.s.requests["b"] = b
        self.c.active_ids = {"b"}
        self.c.pauses["a"] = PausedRequest(9., float("inf"), .8)
        self.c.pauses["a"].resume_by = 9.9
        self.s.waiting = []
        fb = dict(request_id="a", session="s", modality="text", rate=1., received=10.,
            consumed=0., observed_at=10., sequence=2, started=True, complete=False)
        self.write(demand=2., members={"a":dict(rate=1),"b":dict(rate=1)}, feedback={"a":fb})
        with self.c.scheduling_context():self.assertEqual(self.s.running,[b])
        self.assertIsNone(self.c.latched_fault)
        self.assertEqual(self.c.last_mode,"buffer")
        self.assertGreater(self.c.pauses["a"].resume_by, self.now)

    def test_unbuffered_slot_owner_is_not_replaced_by_waiting_resident(self):
        b = request("b", 1)
        self.s.running.append(b)
        self.s.requests["b"] = b
        self.c.active_ids = {"b"}
        self.write(demand=2., members={"a": dict(rate=1), "b": dict(rate=1)})
        with self.c.scheduling_context():
            self.assertEqual(self.s.running, [b])
        self.assertFalse(self.c.pauses)

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
        self.assertEqual(set(self.c.pauses),{"a"})
        self.assertTrue(all(fields["coverage_s"]>0 for event,fields in self.events if event=="client_buffer_pause"))

    def test_unselected_native_resident_with_empty_buffer_is_not_a_policy_pause(self):
        b=request("b",1);self.s.running.append(b);self.s.requests["b"]=b;self.c.active_ids={"a"}
        fb=dict(request_id="b",session="s",modality="text",rate=1.,received=0.,consumed=0.,
                observed_at=10.,sequence=1,started=True,complete=False)
        self.write(demand=2.,members={"a":dict(rate=1),"b":dict(rate=1)},feedback={"b":fb})
        with self.c.scheduling_context():self.assertEqual(self.s.running,[self.a])
        self.assertFalse(self.c.pauses)

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
