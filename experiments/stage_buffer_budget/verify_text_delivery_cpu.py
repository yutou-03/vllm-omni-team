"""CPU checks for conservation, independent intake, and disconnect cleanup."""
import asyncio,importlib.util,json,sys,unittest
from pathlib import Path
from types import ModuleType
trace=ModuleType('vllm_omni.profiling.stage_queue_trace');events=[]
trace.emit_stage_event=lambda name,**kw:events.append(dict(event=name,**kw))
sys.modules['vllm_omni.profiling.stage_queue_trace']=trace
path=Path(__file__).resolve().parents[2]/'vllm_omni/entrypoints/openai/text_delivery_buffer.py'
spec=importlib.util.spec_from_file_location('delivery',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class Tokenizer:
    def encode(self,text,**kw):return list(text)
def packet(text):return 'data: '+json.dumps({'choices':[{'delta':{'content':text}}]})+'\n\n'
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
