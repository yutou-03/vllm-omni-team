"""Verify final-output accounting, control-signal isolation, and buffer depletion."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from .core import ConsumerBuffer
from .gateway import Gateway
from .scheduler import ClientBufferScheduler
from .stream import atomic_json
from .test_core import request
from .virtual_output import VirtualOutput


class VirtualOutputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
        config = dict(modality='text', session='s', arm='buffer', model=tmp.name,
                      execution_telemetry_path=str(root/'execution'), snapshot_path=str(root/'snapshot'),
                      gateway_log=str(root/'events'), buffer_source='server_virtual_output')
        self.g = Gateway(config)
        self.addCleanup(self.g.log.close)
        self.g.w.add('new', 1.)
        self.g.registered_at['new'] = 1.
        self.native = SimpleNamespace(max_num_running_reqs=2, requests={},
                                      _omni_stage_id_for_trace=lambda:0)
        self.now = 10.
        self.a = ClientBufferScheduler(self.native,dict(config, resume_to_client_s=.5),
                                       lambda *a, **kw:None, clock=lambda:self.now)
        self.a.last_snapshot = dict(buffer_source='server_virtual_output')
        self.a.last_mode = 'buffer'

    async def test_accepted_text_batch_to_policy_without_client_feedback(self):
        r = request('new', 0)
        r.client_buffer_test_metadata['sched_request_path'] = 'text'
        self.native.requests[r.request_id] = r
        self.a.record_final_text(r, 4)
        self.a.record_scheduled(SimpleNamespace(num_scheduled_tokens={}))
        self.g.refresh_virtual_output(10.25)
        sample = self.g.feedback['new']
        self.assertEqual((sample.received, sample.consumed), (4., .25))
        snapshot = dict(members={'new':{'rate':1.}}, feedback={'new':asdict(sample)})
        self.assertEqual(self.a.coverage(r,snapshot,10.25),3.75)
        # A large genuine client report cannot replace or duplicate production credit.
        real = ConsumerBuffer(1.)
        real.receive(90.,10.25)
        data = real.feedback(10.25, request_id='new', session='s', modality='text',
                             clock_domain=self.g.clock_domain)
        reply = await self.g.receive_feedback(SimpleNamespace(json=AsyncMock(return_value=data)))
        self.assertEqual(reply.status,200)
        self.assertEqual(self.g.feedback['new'],sample)
        self.g.refresh_virtual_output(15.)
        self.assertEqual(self.g.feedback['new'].coverage(15., .5),0.)
        self.g.config['recovery_ranking'] = 'paper'
        self.assertEqual(self.g.ranking_snapshot(15.)['signed_buffer_s']['new'], -1.)
        self.g.finish_request('new')
        self.g.refresh_virtual_output(16.)
        self.assertFalse(self.g.feedback)
        self.assertFalse(self.g.virtual_output)

    def test_cross_route_output_and_earlier_attempt_have_no_credit(self):
        r = request('new',0)
        r.client_buffer_test_metadata['sched_request_path'] = 'audio'
        self.a.record_final_text(r, 30)
        self.assertFalse(self.a.virtual_text_output)
        stale = VirtualOutput('new',1.,0.,0.)
        stale.produce(30.,.5)
        atomic_json(Path(self.g.config['execution_telemetry_path']),
                    dict(session='s',stage=0,virtual_text_output=[asdict(stale)]))
        self.g.refresh_virtual_output(11.)
        self.assertFalse(self.g.feedback)

    def test_playable_audio_time_does_not_accumulate_consumer_debt(self):
        state = VirtualOutput('audio',.5,10.,10.)
        state.produce(1.,10.)
        self.assertEqual(state.feedback('s','audio',11.).coverage(11.,.5),1.)
        state.produce(.25,19.)
        self.assertEqual(state.feedback('s','audio',19.).coverage(19.,.5),.5)
        self.assertEqual(state.signed_seconds(19.),-6.5)
        state.produce(4., 19.)
        self.assertEqual(state.signed_seconds(19.),1.5)
