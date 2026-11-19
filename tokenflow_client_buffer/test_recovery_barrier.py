"""Mixed admission cannot outrun restoration in the other output route."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from .mixed_gateway import MixedGateway


class MixedRecoveryBarrierTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'tokenizer_config.json').write_text('{"added_tokens_decoder":{}}')
        profile=dict(thinker=dict(raw_capacity=1000.,profile_id='t'),audio=dict(raw_capacity=30.,profile_id='a'),
            audio_thinker_tokens_per_media_s=4.,text_thinker_tokens_per_visible_token=1.)
        (self.root/'profile.json').write_text(json.dumps(profile))
        outputs={m:dict(session=m,modality=m,model=str(self.root),pause_releases_admission=True,
            snapshot_path=str(self.root/(m+'-snapshot.json')),gateway_log=str(self.root/(m+'-gateway.jsonl')),
            execution_telemetry_path=str(self.root/(m+'-execution.json')),upstream='http://127.0.0.1:1',
            native_execution_limit=32) for m in ('text','audio')}
        self.g=MixedGateway(dict(profile=str(self.root/'profile.json'),arm='buffer',outputs=outputs,
            native_execution_limit=32,resource_log=str(self.root/'resources.jsonl')))
        self.addCleanup(self.g.log.close)
        for gateway in self.g.gateways.values():self.addCleanup(gateway.log.close)

    def telemetry(self,modality,recovering):
        Path(self.g.gateways[modality].config['execution_telemetry_path']).write_text(json.dumps(
            dict(session=modality,written_at=time.monotonic(),pause_leases={},recovery_pending=recovering)))

    async def one_tick(self, gateway):
        gateway.stopping=False
        async def stop(delay):
            # A zero-delay fairness yield does not finish draining admission.
            # Stop only when this turn reaches an idle/recovery boundary.
            if delay:
                gateway.stopping = True
        with patch('tokenflow_client_buffer.gateway.asyncio.sleep',stop):
            await gateway.loop()

    async def test_audio_recovery_freezes_text_even_when_shared_capacity_is_feasible(self):
        text=self.g.gateways['text'];text.w.add('new-text',5.);text.ready['new-text']=asyncio.Event()
        self.telemetry('text',False);self.telemetry('audio',True)
        await self.one_tick(text)
        self.assertEqual(text.w.requests['new-text'].state,'waiting')
        self.assertFalse(text.ready['new-text'].is_set())
        self.assertTrue(self.g.gateways['audio'].recovery_pending)
        self.assertIsNone(self.g.book.reason(time.monotonic()))
        self.telemetry('audio',False)
        await self.one_tick(text)
        self.assertEqual(text.w.requests['new-text'].state,'admitted')
        self.assertTrue(text.ready['new-text'].is_set())

    async def test_text_recovery_freezes_audio(self):
        audio=self.g.gateways['audio'];audio.w.add('new-audio',1.);audio.ready['new-audio']=asyncio.Event()
        self.telemetry('text',True);self.telemetry('audio',False)
        await self.one_tick(audio)
        self.assertEqual(audio.w.requests['new-audio'].state,'waiting')
        self.assertFalse(audio.ready['new-audio'].is_set())

    async def test_lost_audio_monitor_freezes_text_without_revoking_safe_pause(self):
        text = self.g.gateways['text']; audio = self.g.gateways['audio']
        text.w.add('new-text', 5.); text.ready['new-text'] = asyncio.Event()
        audio.w.add('old-audio', .5); audio.w.requests['old-audio'].state = 'paused'
        now = time.monotonic()
        audio.accepted_pause_leases['old-audio'] = dict(epoch=now-1., resume_by=now+60.)
        audio.pause_acks['old-audio'] = dict(epoch=now-1., state='paused')
        self.telemetry('text', False)
        await self.one_tick(text)
        self.assertEqual(text.w.requests['new-text'].state, 'waiting')
        self.assertEqual(audio.w.requests['old-audio'].state, 'paused')
        self.assertEqual(audio.pause_acks['old-audio']['state'], 'paused')
        self.assertTrue(audio.recovery_pending)

    async def test_audio_restore_checks_shared_thinker_not_only_audio_capacity(self):
        audio = self.g.gateways['audio']; text = self.g.gateways['text']
        text.w.add('text-owner', 999.); text.w.requests['text-owner'].state = 'admitted'
        audio.w.add('old-audio', .5); audio.w.requests['old-audio'].state = 'paused'
        now = time.monotonic()
        self.assertFalse(audio._restoration_fits('old-audio', now))
        self.g.book.complete_thinker('old-audio')
        self.assertTrue(audio._restoration_fits('old-audio', now))

    async def test_recheck_other_route_between_successive_admissions(self):
        text=self.g.gateways['text']
        for rid in ('first','second'):
            text.w.add(rid,5.);text.ready[rid]=asyncio.Event()
        self.telemetry('text',False);self.telemetry('audio',False)
        original=text.event
        def event(name,**fields):
            original(name,**fields)
            if name=='admitted' and fields['request_id']=='first':self.telemetry('audio',True)
        text.event=event
        await self.one_tick(text)
        self.assertEqual(text.w.requests['first'].state,'admitted')
        self.assertEqual(text.w.requests['second'].state,'waiting')
        self.assertTrue(self.g.gateways['audio'].recovery_pending)


if __name__=='__main__':unittest.main()
