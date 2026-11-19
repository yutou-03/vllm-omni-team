"""Protocol regressions: execute pause, transfer quota, acknowledge restoration."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from .core import CapacityEstimate, CapacityEstimator, ClientFeedback
from .gateway import Gateway
from .scheduler import ClientBufferScheduler
from .test_core import request


class PauseAdmissionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
        self.now = 10.
        self.g = Gateway(dict(modality='text',session='s',model=str(self.root),
            snapshot_path=str(self.root/'snapshot.json'),gateway_log=str(self.root/'gateway.jsonl'),
            execution_telemetry_path=str(self.root/'execution.json'),arm='buffer',
            pause_releases_admission=True,max_inflight=8,native_execution_limit=32))
        self.addCleanup(self.g.log.close)
        self.g.pause_clock = lambda: self.now
        # Isolate the existing pause/restore handshake from the new admission
        # forecast. Tests below explicitly re-enable the full path where needed.
        self.g.w.admission_guard = None
        self.g.est = CapacityEstimate(2.,0.,100.,'text','p','test')
        for rid in ('a','b','c','d'):
            self.g.w.add(rid,1.); self.g.registered_at[rid] = 1.
        for rid in ('a','b'):
            self.assertEqual(self.g.w.admit_next(self.g.est,self.now),rid)
        self.a, self.b = request('a',0), request('b',1)
        self.s = NS(running=[self.a,self.b],waiting=[],requests={'a':self.a,'b':self.b},
                    max_num_running_reqs=32,kv_cache_manager=NS(),_omni_stage_id_for_trace=lambda:0)
        self.events = []
        self.adapter = ClientBufferScheduler(self.s,dict(snapshot_path=self.g.config['snapshot_path'],
            execution_telemetry_path=self.g.config['execution_telemetry_path'],resume_to_client_s=.5,
            session='s',modality='text'),lambda event,**kw:self.events.append((event,kw)),clock=lambda:self.now)
        self.feedback('a',10.)
        self.feedback('b',.1)

    def feedback(self,rid,units,started=True):
        self.g.feedback[rid] = ClientFeedback(rid,'s','text',1.,units,0.,self.now,1,started,False)

    def snapshot(self):
        d = self.g.w.snapshot(self.g.est,self.now)
        d.update(schema_version=1,session='s',modality='text',written_at=self.now,
                 capacity_valid_until=100.,feedback={rid:asdict(f) for rid,f in self.g.feedback.items()},
                 **self.g.pause_snapshot(self.now))
        Path(self.g.config['snapshot_path']).write_text(json.dumps(d))
        return d

    def step(self):
        self.snapshot()
        with self.adapter.scheduling_context():
            selected = {r.request_id:1 for r in self.s.running}
        self.adapter.record_scheduled(NS(num_scheduled_tokens=selected))
        return selected

    def test_full_guard_allows_client_safe_pause_admit_first_output_and_restore(self):
        self.g.w.admission_guard = self.g.admission_safety
        self.feedback('b', 2.)  # also protect the unpaused stream from new prefill
        self.assertEqual(set(self.step()), {'b'})
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.admit_next(self.g.est, self.now), 'c')
        c = request('c', 2); self.s.running.append(c); self.s.requests['c'] = c
        self.now = 10.1; self.feedback('a', 9.9); self.feedback('b', 1.9)
        self.assertIn('c', self.step())
        self.feedback('c', 2.)
        self.g.w.complete('b'); self.s.running.remove(self.b); del self.s.requests['b']
        self.now = 10.2; self.feedback('a', 9.8)
        self.step(); self.g.process_pause_leases(self.now)
        self.now = 10.3
        self.assertIn('a', self.step())
        self.assertNotIn('a', self.adapter.pauses)

    def test_gateway_waiter_causes_pause_below_32_and_new_admission_before_completion(self):
        # No engine waiter exists: the old implementation could never fund C.
        self.assertFalse(self.s.waiting)
        self.assertIsNone(self.g.w.admit_next(self.g.est,self.now))
        self.assertEqual(set(self.step()),{'b'})
        self.assertEqual(self.g.w.demand,2.)  # not released before confirmation
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.requests['a'].state,'paused')
        self.assertEqual(self.g.w.demand,1.)
        self.assertEqual(self.g.w.admit_next(self.g.est,self.now),'c')
        self.assertEqual(self.g.w.demand,2.)
        self.assertEqual(len(self.g.w.outstanding),3)
        self.assertFalse(any(r.state=='finished' for r in self.g.w.requests.values()))
        c=request('c',2);self.s.running.append(c);self.s.requests['c']=c
        self.now=10.1;self.feedback('a',9.9)
        self.assertEqual(set(self.step()),{'b','c'})
        self.assertIn('a',self.adapter.pauses)
        self.feedback('c',.1)  # C has now reached first output
        self.g.w.complete('b');self.s.running.remove(self.b);del self.s.requests['b']
        self.now=10.2;self.feedback('a',9.8)
        self.assertEqual(set(self.step()),{'c'})  # resume waits for quota ACK
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,2.)
        self.assertEqual(self.g.w.mode,'buffer')
        self.assertIsNone(self.g.w.admit_next(self.g.est,self.now))
        self.now=10.3
        self.assertIn('a',self.step())
        self.assertNotIn('a',self.adapter.pauses)
        self.assertFalse(any(k=='client_buffer_resume_ack_timeout' for k,v in self.events))

    def test_resume_exchanges_active_quota_even_with_free_gpu_slots(self):
        self.feedback('b',8.)
        self.step();self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.admit_next(self.g.est,self.now),'c')
        c=request('c',2);self.s.running.append(c);self.s.requests['c']=c
        self.now=10.1;self.feedback('a',1.1);self.feedback('b',7.9);self.feedback('c',.1)
        self.assertEqual(set(self.step()),{'c'})
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.requests['a'].state,'admitted')
        self.assertEqual(self.g.w.requests['b'].state,'paused')
        self.assertEqual(self.g.w.demand,2.)
        self.assertEqual(self.g.w.mode,'buffer')
        self.now=10.2;self.feedback('a',1.);self.feedback('b',7.8)
        self.assertEqual(set(self.step()),{'a','c'})
        self.assertNotIn('a',self.adapter.pauses)
        self.assertIn('b',self.adapter.pauses)

    def test_passed_replenishment_deadline_never_bypasses_quota_ack(self):
        self.step(); self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.admit_next(self.g.est,self.now),'c')
        c=request('c',2);self.s.running.append(c);self.s.requests['c']=c
        self.now=10.1
        # The replenishment estimate moves the resume deadline into the past.
        # This is not a control timeout: the return has not been requested yet.
        self.feedback('a',.2);self.feedback('b',.1);self.feedback('c',.1)
        self.assertNotIn('a',self.step())
        self.assertIn('a',self.adapter.pauses.requested_resumes)
        self.assertFalse(any(k=='client_buffer_resume_ack_timeout' for k,v in self.events))
        self.g.process_pause_leases(self.now)
        self.now=10.15
        self.assertIn('a',self.step())
        self.assertEqual(self.g.w.requests['a'].state,'admitted')
        self.assertFalse(any(k=='client_buffer_resume_ack_timeout' for k,v in self.events))

    def test_forced_restoration_overload_freezes_new_admission(self):
        self.step();self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.admit_next(self.g.est,self.now),'c')
        path=Path(self.g.config['execution_telemetry_path'])
        t=json.loads(path.read_text());t['pause_leases']['a']['state']='resume_requested'
        t['pause_leases']['a']['resume_by']=self.now
        path.write_text(json.dumps(t));self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,3.)
        self.assertEqual(self.g.w.mode,'fcfs')
        self.assertEqual(self.g.w.reason,'overload')
        self.assertIsNone(self.g.w.admit_next(self.g.est,self.now))

    def test_capacity_fallback_still_restores_due_pause_before_deadline(self):
        self.step(); self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.admit_next(self.g.est,self.now),'c')
        c=request('c',2);self.s.running.append(c);self.s.requests['c']=c
        self.now=10.1
        self.g.est=CapacityEstimate(1.5,0.,100.,'text','p','test')
        self.feedback('a',1.0);self.feedback('b',.1);self.feedback('c',.1)
        self.assertNotIn('a',self.step())  # handshake, not an uncharged resume
        self.assertIn('a',self.adapter.pauses.requested_resumes)
        self.assertTrue(self.adapter.recovery_pending)
        self.g.process_pause_leases(self.now)
        self.assertTrue(self.g.recovery_pending)
        self.assertEqual(self.g.w.requests['a'].state,'admitted')
        self.assertIsNone(self.g.w.admit_next(self.g.est,self.now))
        self.now=10.15;self.feedback('a',.95)
        self.assertIn('a',self.step())
        self.assertNotIn('a',self.adapter.pauses)
        self.assertFalse(any(k=='client_buffer_resume_ack_timeout' for k,v in self.events))

    def test_fcfs_recovery_swaps_buffered_owner_without_exceeding_native_cap(self):
        self.feedback('b',8.)
        self.step();self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.admit_next(self.g.est,self.now),'c')
        c=request('c',2);self.s.running.append(c);self.s.requests['c']=c
        self.adapter.cap=self.adapter.resident_cap=self.s.max_num_running_reqs=2
        self.adapter.active_ids={'b','c'}
        self.now=10.1;self.g.est=CapacityEstimate(1.5,0.,100.,'text','p','test')
        self.feedback('a',1.);self.feedback('b',7.9);self.feedback('c',.1)
        self.assertEqual(set(self.step()),{'c'})
        self.assertIn('b',self.adapter.pauses)
        self.g.process_pause_leases(self.now)
        self.now=10.15;self.feedback('a',.95);self.feedback('b',7.85)
        self.assertEqual(set(self.step()),{'a','c'})
        self.assertIn('b',self.adapter.pauses)

    def test_native_waiter_cannot_steal_slot_during_resume_ack(self):
        self.feedback('b',8.)
        self.step();self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.admit_next(self.g.est,self.now),'c')
        c=request('c',2);self.s.running.append(c);self.s.requests['c']=c
        d=request('d',3);d.status.name='WAITING';self.s.waiting=[d];self.s.requests['d']=d
        self.adapter.cap=self.adapter.resident_cap=self.s.max_num_running_reqs=2
        self.adapter.active_ids={'b','c'}
        self.now=10.1;self.feedback('a',1.);self.feedback('b',7.9);self.feedback('c',.1)
        self.snapshot()
        with self.adapter.scheduling_context():
            self.assertEqual([r.request_id for r in self.s.running],['c'])
            self.assertFalse(self.s.waiting)  # native cannot consume return vacancy
        self.assertEqual(self.s.waiting,[d])
        # A second step before the ACK must keep the same reservation, even
        # when the due loop recognizes an exchange already in flight.
        self.now=10.12;self.feedback('a',.98);self.feedback('b',7.88)
        self.snapshot()
        with self.adapter.scheduling_context():
            self.assertFalse(self.s.waiting)
        self.assertEqual(self.s.waiting,[d])

    def test_chunk_preempted_pause_in_waiting_needs_quota_ack_before_native_fill(self):
        self.feedback('b',8.)
        self.step();self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.admit_next(self.g.est,self.now),'c')
        c=request('c',2);self.s.running.append(c);self.s.requests['c']=c
        self.s.running.remove(self.a);self.a.status.name='PREEMPTED';self.s.waiting=[self.a]
        self.now=10.1;self.feedback('a',1.);self.feedback('b',7.9);self.feedback('c',.1)
        self.snapshot()
        with self.adapter.scheduling_context():
            self.assertFalse(self.s.waiting)
            selected={r.request_id:1 for r in self.s.running}
        self.adapter.record_scheduled(NS(num_scheduled_tokens=selected))
        self.assertIn('a',self.adapter.pauses.requested_resumes)
        self.assertIn('a',self.adapter.pauses)
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.requests['a'].state,'admitted')
        self.now=10.15;self.feedback('a',.95);self.feedback('b',7.85)
        self.snapshot()
        with self.adapter.scheduling_context():
            self.assertEqual(self.s.waiting,[self.a])
            r=self.s.waiting.pop(0);r.status.name='RUNNING';self.s.running.append(r)
            selected={r.request_id:1 for r in self.s.running}
        self.adapter.record_scheduled(NS(num_scheduled_tokens=selected))
        self.assertIn('a',selected)
        self.assertNotIn('a',self.adapter.pauses)

    def test_startup_longer_than_safe_lease_blocks_new_admission(self):
        self.g.w.admission_guard = self.g.admission_safety
        self.g.config['admission_startup_s']=5.5
        self.feedback('a',5.)
        self.assertEqual(set(self.step()),{'a','b'})
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,2.)
        self.assertFalse(self.adapter.pauses)
        self.assertIsNone(self.g.w.admit_next(self.g.est,self.now))

    def test_gateway_leaves_early_warning_interval_for_atomic_exchange(self):
        self.step();self.g.process_pause_leases(self.now)
        deadline=self.g.accepted_pause_leases['a']['resume_by']
        self.now=deadline-.4
        self.feedback('a',.4+.75)
        path=Path(self.g.config['execution_telemetry_path']);t=json.loads(path.read_text())
        t['written_at']=self.now;t['pause_leases']['a']['not_scheduled_at']=self.now
        path.write_text(json.dumps(t));self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.requests['a'].state,'paused')
        self.assertEqual(self.g.w.demand,1.)

    def test_only_new_client_credit_can_renew_pause_deadline(self):
        self.step();self.g.process_pause_leases(self.now)
        old_deadline=self.g.accepted_pause_leases['a']['resume_by']
        self.now=11.;self.feedback('a',13.)
        path=Path(self.g.config['execution_telemetry_path']);t=json.loads(path.read_text())
        t['written_at']=self.now;t['pause_leases']['a'].update(not_scheduled_at=self.now,resume_by=old_deadline+4.)
        path.write_text(json.dumps(t));self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.accepted_pause_leases['a']['resume_by'],old_deadline+4.)
        t['pause_leases']['a']['resume_by']=100000.;t['receiver_ready_inventory_units']=1000000
        path.write_text(json.dumps(t));self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.accepted_pause_leases['a']['resume_by'],old_deadline+4.)

    def test_no_feedback_cannot_release_admission_even_with_telemetry(self):
        self.step();self.g.feedback.clear()
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,2.)
        self.assertIsNone(self.g.w.admit_next(self.g.est,self.now))

    def test_stale_scheduler_telemetry_cannot_release_admission(self):
        self.step();self.now=11.;self.feedback('a',9.)
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,2.)

    def test_old_case_pause_cannot_apply_to_new_request_with_same_id(self):
        self.step();self.g.registered_at['a']=10.01
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,2.)

    def test_missing_telemetry_preserves_safe_lease_and_freezes_admission(self):
        self.step();self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,1.)
        path=Path(self.g.config['execution_telemetry_path']);old=path.read_text();path.unlink()
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,1.)
        self.assertTrue(self.g.recovery_pending)
        path.write_text(old);self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,1.)
        self.assertEqual(self.g.pause_acks['a']['state'],'paused')
        self.assertFalse(self.g.recovery_pending)

    def test_r31_long_steps_do_not_restore_22_or_18_safe_pauses(self):
        for count, gap in ((22, .567), (18, .766)):
            with self.subTest(paused=count, gap=gap):
                self.g.w.requests.clear()
                self.g.accepted_pause_leases.clear()
                self.g.pause_acks.clear()
                self.g.feedback.clear()
                self.g.est = CapacityEstimate(30.657, 0., 1000., 'text', 'p', 'test')
                leases = {}
                for index in range(61 + count):
                    rid = str(index)
                    self.g.w.add(rid, .5)
                    self.g.w.requests[rid].state = 'admitted'
                    self.g.registered_at[rid] = 1.
                    if index < count:
                        self.g.feedback[rid] = ClientFeedback(rid, 's', 'text', .5,
                            60., 0., self.now, 1, True, False)
                        leases[rid] = dict(epoch=self.now, state='paused',
                            resume_by=self.now+100., resume_guard_s=1., not_scheduled_at=self.now)
                Path(self.g.config['execution_telemetry_path']).write_text(json.dumps(
                    dict(session='s', written_at=self.now, pause_leases=leases)))
                self.g.process_pause_leases(self.now)
                self.assertEqual(self.g.w.demand, 30.5)
                self.now += gap
                self.g.process_pause_leases(self.now)
                self.assertEqual(self.g.w.demand, 30.5)
                self.assertEqual(len(self.g.accepted_pause_leases), count)
                self.assertTrue(self.g.recovery_pending)
                self.assertEqual(self.g.w.mode, 'buffer')

    def test_long_step_then_fresh_telemetry_keeps_borrower_running(self):
        self.step(); self.g.process_pause_leases(self.now)
        self.g.w.admit_next(self.g.est, self.now)
        c = request('c', 2); self.s.running.append(c); self.s.requests['c'] = c
        deadline = self.g.accepted_pause_leases['a']['resume_by']
        self.now += .567
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.accepted_pause_leases['a']['resume_by'], deadline)
        self.assertTrue(self.g.recovery_pending)
        self.feedback('a', 9.433); self.feedback('b', .1); self.feedback('c', .1)
        self.assertEqual(set(self.step()), {'b', 'c'})
        self.g.process_pause_leases(self.now)
        self.assertFalse(self.g.recovery_pending)
        self.assertEqual(self.g.w.demand, 2.)
        self.assertEqual(self.g.pause_acks['a']['state'], 'paused')

    def test_return_is_charged_after_confirmed_exchange_without_transient_overload(self):
        self.feedback('b', 8.)
        self.step(); self.g.process_pause_leases(self.now)
        self.g.w.admit_next(self.g.est, self.now)
        c = request('c', 2); self.s.running.append(c); self.s.requests['c'] = c
        self.now = 10.1
        self.feedback('a', 1.1); self.feedback('b', 7.9); self.feedback('c', .1)
        self.step()
        demand_events = []
        original = self.g.event
        def event(name, **fields):
            if name in ('pause_admission_released', 'pause_admission_restored'):
                demand_events.append((name, self.g.w.demand))
            original(name, **fields)
        self.g.event = event
        self.g.process_pause_leases(self.now)
        self.assertEqual(demand_events, [('pause_admission_released', 1.),
                                        ('pause_admission_restored', 2.)])

    def test_read_timestamp_race_uses_time_after_read(self):
        self.step()
        path = Path(self.g.config['execution_telemetry_path'])
        t = json.loads(path.read_text()); t['written_at'] = self.now+.001
        path.write_text(json.dumps(t))
        self.g.pause_clock = lambda: self.now+.002
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.requests['a'].state, 'paused')
        self.assertEqual(self.g.pause_telemetry_status, 'fresh')

    def test_unfunded_early_return_defers_but_deadline_forces_restoration(self):
        self.step(); self.g.process_pause_leases(self.now)
        self.g.w.admit_next(self.g.est, self.now)
        path = Path(self.g.config['execution_telemetry_path'])
        t = json.loads(path.read_text()); t['pause_leases']['a']['state'] = 'resume_requested'
        path.write_text(json.dumps(t)); self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand, 2.)
        self.assertEqual(self.g.pause_acks['a']['state'], 'resume_deferred')
        self.assertTrue(self.g.recovery_pending)
        self.now = self.g.accepted_pause_leases['a']['resume_by']+.01
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand, 3.)
        self.assertEqual(self.g.pause_acks['a']['state'], 'active')
        self.assertEqual(self.g.w.reason, 'overload')

    def test_deferred_return_retries_safe_exchange_without_ack_timeout(self):
        self.step(); self.g.process_pause_leases(self.now)
        self.g.w.admit_next(self.g.est, self.now)
        c = request('c', 2); self.s.running.append(c); self.s.requests['c'] = c
        self.feedback('c', .1)
        self.now = 10.1
        self.feedback('a', 9.9); self.feedback('b', .1)
        self.adapter.pauses['a'].request_resume(self.now)
        self.adapter.pauses['a'].ack_wait_since = 9.
        self.g.pause_acks['a']['state'] = 'resume_deferred'
        self.assertNotIn('a', self.step())
        self.assertIsNone(self.adapter.latched_fault)
        self.now = 10.2
        self.feedback('a', 9.8); self.feedback('b', 8.); self.feedback('c', .1)
        self.step(); self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.requests['a'].state, 'admitted')
        self.assertEqual(self.g.w.requests['b'].state, 'paused')
        self.assertEqual(self.g.w.demand, 2.)
        self.now = 10.3
        self.assertIn('a', self.step())
        self.assertIsNone(self.adapter.latched_fault)

    def test_finished_pause_does_not_leave_a_stale_admission_barrier(self):
        self.step(); self.g.process_pause_leases(self.now)
        self.g.w.complete('a'); self.g.w.complete('b')
        Path(self.g.config['execution_telemetry_path']).unlink()
        self.g.process_pause_leases(self.now)
        self.assertFalse(self.g.accepted_pause_leases)
        self.assertFalse(self.g.recovery_pending)

    def test_expired_lease_restores_demand_without_scheduler_request(self):
        self.step();self.g.process_pause_leases(self.now)
        self.now=self.g.accepted_pause_leases['a']['resume_by']+.01
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,2.)
        self.assertEqual(self.g.pause_acks['a']['state'],'active')

    def test_expired_capacity_with_retained_pause_falls_back_and_restores(self):
        self.step(); self.g.process_pause_leases(self.now)
        self.g.w.admit_next(self.g.est, self.now)
        c = request('c', 2); self.s.running.append(c); self.s.requests['c'] = c
        self.g.est = None
        self.now = 10.1
        self.feedback('a', 9.9); self.feedback('b', .1); self.feedback('c', .1)
        self.assertNotIn('a', self.step())
        self.assertEqual(self.adapter.last_mode, 'fcfs')
        self.assertIsNone(self.adapter.latched_fault)
        self.now = 10.2
        self.feedback('a', .2); self.feedback('b', .1); self.feedback('c', .1)
        self.assertNotIn('a', self.step())
        self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.pause_acks['a']['state'], 'active')
        self.assertTrue(self.g.recovery_pending)
        # Gateway.loop checks recovery_pending before calling admit_next.
        self.assertEqual(self.g.w.requests['d'].state, 'waiting')
        self.now = 10.25
        self.assertIn('a', self.step())

    def test_paused_requests_keep_memory_accounting(self):
        self.g.w.max_inflight=2
        self.step();self.g.process_pause_leases(self.now)
        self.assertEqual(self.g.w.demand,1.)
        self.assertIsNone(self.g.w.admit_next(self.g.est,self.now))

    def test_native_fcfs_does_not_use_pause_quota_protocol(self):
        self.g.config['arm']='fcfs'
        self.assertEqual(self.g.pause_snapshot(self.now),{})

    def test_default_capacity_has_no_extra_admission_haircut(self):
        estimate=CapacityEstimator('audio').seed(30.,0.,'p')
        self.assertEqual(estimate.admission_bound,30.)

    def test_zero_execution_window_does_not_crash_gateway(self):
        self.g.window_start=self.g.last_capacity_check=0.
        self.g.previous_telemetry=dict(steps=0,full_steps=0,held_steps=0)
        Path(self.g.config['execution_telemetry_path']).write_text(json.dumps(dict(
            steps=0,full_steps=0,held_steps=0,written_at=10.,session='s')))
        self.g.update_capacity(10.)
        self.assertEqual(self.g.est.gamma,2.)
        event=json.loads(Path(self.g.config['gateway_log']).read_text().splitlines()[-1])
        self.assertEqual(event['saturation_basis'],'no_valid_execution')



    def test_rate_deficit_with_buffer_credit_does_not_force_an_unnecessary_pause(self):
        from .admission import DeliveryHistory
        self.g.w.admission_guard = self.g.admission_safety
        self.g.est = CapacityEstimate(30., 0., 100., 'text', 'p', 'test')
        self.g.config['admission_startup_s'] = 5.5
        self.feedback('b', .1)
        history = self.g.delivery_history['a'] = DeliveryHistory()
        for i in range(3):
            f = ClientFeedback('a', 's', 'text', 1., 20.+.8*(i+1), .8*(i+1),
                self.now-2+i, i+1, True, False)
            history.observe(f, {'text': 2.})
        self.g.feedback['a'] = f
        self.assertEqual(self.g.w.admit_next(self.g.est, self.now), 'c')
        self.assertEqual(self.g.w.requests['a'].state, 'admitted')
        self.assertEqual(self.g.w.demand, 3.)
        self.assertFalse(self.g.accepted_pause_leases)
        self.assertFalse(self.adapter.pauses)

if __name__=='__main__':unittest.main()
