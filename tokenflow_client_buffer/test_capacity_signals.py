import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from .capacity_signals import CapacityExecutionSignals, execution_window_evidence
from .core import CapacityEstimator


def telemetry(seconds, *, full=0, gpu=True, pressure=True, budget=False):
    return dict(steps=int(seconds*10), full_steps=full, buffer_steps=int(seconds*10),
        held_steps=0, capacity_observed_s=seconds,
        capacity_hardware_valid_s=seconds if gpu else 0.,
        capacity_backend_pressure_s=seconds if gpu and pressure else 0.,
        capacity_budget_pressure_s=seconds if budget else 0.,
        written_at=seconds, session='audio')


class CapacitySignalsTests(unittest.TestCase):
    def test_hardware_pressure_below_32_can_update_downward(self):
        evidence=execution_window_evidence(telemetry(8.),telemetry(0.),8.)
        self.assertTrue(evidence['saturated'])
        self.assertEqual(evidence['saturation_basis'],'hardware_busy_with_runnable_backlog')
        e=CapacityEstimator('audio',alpha=1.,min_window_s=8.);e.seed(30.,0.,'p')
        e.observe(generated_units=160.,duration_s=8.,now=8.,
            saturated=evidence['saturated'],profile_id='p')
        self.assertEqual(e.estimate.gamma,20.)

    def test_busy_scheduled_work_updates_without_full_batch_or_extra_backend_waiters(self):
        previous=telemetry(0.,pressure=False)
        current=telemetry(8.,pressure=False)
        current['capacity_busy_work_s']=8.
        evidence=execution_window_evidence(current,previous,8.)
        self.assertEqual(evidence['saturation_basis'],'hardware_busy_with_scheduled_work')
        e=CapacityEstimator('audio',alpha=1.,min_window_s=8.);e.seed(30.,0.,'p')
        e.observe(generated_units=160.,duration_s=8.,now=8.,
                  saturated=evidence['saturated'],profile_id='p')
        self.assertEqual(e.estimate.gamma,20.)

    def test_full_32_without_independent_evidence_cannot_lower_capacity(self):
        evidence=execution_window_evidence(telemetry(8.,full=80,gpu=False),telemetry(0.),8.)
        self.assertFalse(evidence['saturated'])
        e=CapacityEstimator('audio',alpha=1.,min_window_s=8.);old=e.seed(30.,0.,'p')
        self.assertIs(e.observe(generated_units=160.,duration_s=8.,now=8.,
            saturated=evidence['saturated'],profile_id='p'),old)

    def test_hardware_busy_without_backend_backlog_cannot_self_contract(self):
        evidence=execution_window_evidence(telemetry(8.,pressure=False),telemetry(0.),8.)
        self.assertFalse(evidence['saturated'])
        evidence=execution_window_evidence(telemetry(8.,gpu=False,budget=True),telemetry(0.),8.)
        self.assertTrue(evidence['saturated'])
        self.assertEqual(evidence['saturation_basis'],'token_budget_with_runnable_backlog')

    def test_fcfs_output_recovery_and_policy_cost_remain_measured(self):
        e=CapacityEstimator('text',alpha=1.);e.seed(100.,0.,'p')
        e.observe(generated_units=100.,duration_s=2.,now=2.,saturated=True,profile_id='p',policy='buffer')
        self.assertEqual(e.estimate.gamma,50.)
        e.observe(generated_units=160.,duration_s=2.,now=4.,saturated=False,profile_id='p',policy='fcfs')
        self.assertEqual(e.estimate.gamma,80.)
        e.observe(generated_units=120.,duration_s=2.,now=6.,saturated=True,profile_id='p',policy='buffer')
        self.assertEqual(e.estimate.gamma,60.)

    def test_sampler_excludes_policy_holds_and_input_blocked_requests(self):
        class Sampler:
            def __init__(self):self.calls=0
            def sample(self,**kw):
                self.calls+=1
                return NS(ratio=.95,sample_age_ms=0,unavailable_reason=None)
        def request(rid,status):
            return NS(request_id=rid,status=NS(name=status),num_computed_tokens=101,num_prompt_tokens=10)
        a,p,b=request('a','RUNNING'),request('p','RUNNING'),request('b','WAITING_FOR_CHUNK')
        s=NS(requests={r.request_id:r for r in (a,p,b)},running=[a,p],waiting=[b],max_num_scheduled_tokens=128)
        sampler=Sampler();signals=CapacityExecutionSignals(s,{},samplers=[sampler])
        output=NS(num_scheduled_tokens={'a':1})
        start=signals.record(output,0.,paused_ids={'p'})
        for i in range(1,11):end=signals.record(output,i/10,paused_ids={'p'})
        self.assertEqual(sampler.calls,3)
        self.assertEqual(end['capacity_runnable_backend_waiters'],0)
        self.assertEqual(end['capacity_mean_context_tokens'],100)
        self.assertEqual(execution_window_evidence(end,start,1.)['saturation_basis'],
                         'hardware_busy_with_scheduled_work')
        b.status.name='WAITING'
        for i in range(11,31):end=signals.record(output,i/10,paused_ids={'p'})
        self.assertEqual(end['capacity_runnable_backend_waiters'],1)
        self.assertGreater(end['capacity_backend_pressure_s'],1.)

    def test_idle_gap_does_not_extend_stale_gpu_busy_sample(self):
        class Sampler:
            def sample(self,**kw):return NS(ratio=1.,sample_age_ms=0,unavailable_reason=None)
        r=NS(request_id='a',status=NS(name='RUNNING'),num_computed_tokens=10,num_prompt_tokens=2)
        w=NS(request_id='w',status=NS(name='WAITING'))
        s=NS(requests={'a':r},running=[r],waiting=[w],max_num_scheduled_tokens=32)
        signal=CapacityExecutionSignals(s,{},samplers=[Sampler()]);output=NS(num_scheduled_tokens={'a':1})
        start=signal.record(output,0.);end=signal.record(output,10.)
        self.assertFalse(execution_window_evidence(end,start,10.)['saturated'])
        self.assertEqual(end['capacity_hardware_valid_s'],0.)

    def test_mixed_audio_silent_underfilled_window_keeps_denominator(self):
        from .mixed_gateway import MixedGateway
        from .mixed_core import MixedBook
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'telemetry.json'
            e=CapacityEstimator('audio',alpha=1.,min_window_s=8.)
            audio=e.seed(20.,0.,'a');other=CapacityEstimator('text').seed(1000.,0.,'t')
            g=MixedGateway.__new__(MixedGateway);g.estimators={'audio':e};g.book=MixedBook(other,audio,4.)
            g.windows={'audio':dict(start=0.,previous=telemetry(0.),units=0.,last_check=0.)}
            g.gateways={'audio':NS(config={'execution_telemetry_path':str(path)},session='audio',window_units=0.)}
            events=[];g.event=lambda name,**kw:events.append(kw)
            path.write_text(json.dumps(telemetry(8.,gpu=False)))
            g.observe('audio',8.)
            self.assertEqual(g.windows['audio']['start'],0.)
            self.assertTrue(events[-1]['deferred_empty_window'])
            g.gateways['audio'].window_units=300.
            path.write_text(json.dumps(telemetry(10.,gpu=False)))
            g.observe('audio',10.)
            self.assertEqual(e.estimate.gamma,30.)  # 300/10, never 300/2.
            self.assertEqual(events[-1]['observed_output_per_s'],30.)
            self.assertFalse(events[-1]['saturated'])


if __name__=='__main__':unittest.main()
