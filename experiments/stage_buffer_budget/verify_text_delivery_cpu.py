"""CPU checks for conservation, independent intake, and disconnect cleanup."""
import asyncio,importlib.util,json,sys,unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
trace=ModuleType('vllm_omni.profiling.stage_queue_trace');events=[]
trace.emit_stage_event=lambda name,**kw:events.append(dict(event=name,**kw))
sys.modules['vllm_omni.profiling.stage_queue_trace']=trace
path=Path(__file__).resolve().parents[2]/'vllm_omni/entrypoints/openai/text_delivery_buffer.py'
spec=importlib.util.spec_from_file_location('delivery',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Tokenizer:
    def encode(self,text,**kw):return list(text)
def packet(text):return 'data: '+json.dumps({'choices':[{'delta':{'content':text}}]})+'\n\n'
class ControlledClock:
    """Advance only requested delivery delays; keep asyncio scheduling real."""
    def __init__(self):
        self.now = 100.
        self.delays = []

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    async def sleep(self, seconds):
        self.delays.append(seconds)
        self.advance(seconds)
        await asyncio.sleep(0)

    def asyncio_proxy(self):
        return SimpleNamespace(
            Queue=asyncio.Queue, create_task=asyncio.create_task,
            CancelledError=asyncio.CancelledError, sleep=self.sleep)


class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_wire_and_receives_while_delivery_waits(self):
        events.clear();records=[packet('ab'),packet('cd'),packet('ef'),'data: [DONE]\n\n']
        async def source():
            for p in records:yield p
        out=[p async for p in mod.pace_text_stream(source(),tokenizer=Tokenizer(),request_id='r',config={'tokens_per_s':100})]
        self.assertEqual(out,records)
        final=events[-1]
        self.assertEqual(final['received_wire_sha256'],final['released_wire_sha256'])
        self.assertEqual(final['received_text_sha256'],final['released_text_sha256'])
        self.assertEqual(final['buffered_chars'],0)
        self.assertTrue(any(e['buffered_tokens']>=4 for e in events))
    async def test_ready_producer_does_not_starve_delivery(self):
        produced=[]
        async def source():
            for i in range(200):
                produced.append(i)
                yield packet('a')
        stream=mod.pace_text_stream(source(),tokenizer=Tokenizer(),request_id='fair',config={'tokens_per_s':1000})
        await anext(stream)
        self.assertLess(len(produced),20)
        await stream.aclose()

    async def test_multitoken_packets_follow_cumulative_token_time(self):
        events.clear()
        # Different packet sizes must consume different amounts of the same
        # token budget. Wire packet boundaries and metadata must stay intact.
        records = [packet('ab'), packet('cde'), packet('fghi'), packet('j'),
                   'data: {"choices": [], "usage": {"completion_tokens": 10}}\n\n',
                   'data: [DONE]\n\n']
        async def source():
            for record in records:
                yield record

        clock = ControlledClock()
        output, delivered_at = [], []
        with patch.object(mod, 'time', clock), patch.object(mod, 'asyncio', clock.asyncio_proxy()):
            async for record in mod.pace_text_stream(
                    source(), tokenizer=Tokenizer(), request_id='multitoken',
                    config={'tokens_per_s': 4.8, 'delivery_lead_tokens': 1}):
                output.append(record)
                delivered_at.append(clock.monotonic() - 100.)

        self.assertEqual(output, records)
        # With one token of lead, the second packet follows the first token's
        # consumption; the later intervals reflect packets of 3 and 4 tokens.
        expected_times = [0., 1. / 4.8, 4. / 4.8, 8. / 4.8]
        for actual, expected in zip(delivered_at[:4], expected_times):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(delivered_at[4:], [delivered_at[3]] * 2)
        self.assertEqual(events[-1]['released_tokens'], 10)
        self.assertEqual(events[-1]['buffered_tokens'], 0)
        self.assertEqual(events[-1]['received_wire_sha256'], events[-1]['released_wire_sha256'])

    async def test_late_delivery_catches_up_and_exposes_inventory_debt(self):
        events.clear()
        ready = asyncio.Event()
        records = [packet('ab'), packet('cd'), packet('ef'), packet('gh'),
                   'data: [DONE]\n\n']
        async def source():
            yield records[0]
            await ready.wait()
            for record in records[1:]:
                yield record

        clock = ControlledClock()
        output, delivered_at = [], []
        with patch.object(mod, 'time', clock), patch.object(mod, 'asyncio', clock.asyncio_proxy()):
            stream = mod.pace_text_stream(
                source(), tokenizer=Tokenizer(), request_id='late',
                config={'tokens_per_s': 4.8, 'delivery_lead_tokens': 1})
            output.append(await anext(stream))
            delivered_at.append(clock.monotonic() - 100.)
            # Simulate an event-loop or source delay without spending wall time.
            # Two subsequent packets are already due on the original clock.
            clock.advance(.9)
            ready.set()
            async for record in stream:
                output.append(record)
                delivered_at.append(clock.monotonic() - 100.)

        self.assertEqual(output, records)
        for actual, expected in zip(delivered_at[:4], [0., .9, .9, 5. / 4.8]):
            self.assertAlmostEqual(actual, expected)
        # The receiver must expose catch-up demand before releasing it, so the
        # pause controller can avoid crediting overdue tokens as safe stock.
        debt_events = [e for e in events if e['action'] == 'received'
                       and e['released_tokens'] == 2 and e['buffered_tokens'] >= 2]
        self.assertTrue(debt_events)
        self.assertAlmostEqual(debt_events[0]['pacing_debt_tokens'], 3.32)
        self.assertAlmostEqual(events[-1]['pacing_debt_tokens'], 0.)
        self.assertEqual(events[-1]['received_text_sha256'], events[-1]['released_text_sha256'])

    async def test_disconnect_cancels_full_producer_queue(self):
        closed=[]
        async def source():
            try:
                for _ in range(100):yield packet('a')
            finally:closed.append(True)
        stream=mod.pace_text_stream(source(),tokenizer=Tokenizer(),request_id='r',config={'tokens_per_s':4.8,'max_events':1})
        await anext(stream);await asyncio.sleep(.01)
        await asyncio.wait_for(stream.aclose(),timeout=.5)
        self.assertTrue(closed)
    async def test_bound_failure_is_not_silent_truncation(self):
        async def source():yield packet('abcd')
        with self.assertRaisesRegex(RuntimeError,'bound exceeded'):
            async for _ in mod.pace_text_stream(source(),tokenizer=Tokenizer(),request_id='r',config={'max_chars':2}):pass
if __name__=='__main__':unittest.main()
