"""Standalone CPU contract checks: python3 tests/profiling/test_stage_study_cpu.py.

Load only instrumentation modules and AST-extracted CPU branches; no vLLM,
CUDA initialization, model import, or model inference is performed.
"""
from __future__ import annotations
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_modules():
    names = ('vllm_omni', 'vllm_omni.profiling', 'vllm_omni.profiling.nvtx')
    saved = {name: sys.modules.get(name) for name in names}
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules['vllm_omni.profiling.nvtx'].nvtx_range = lambda *a, **kw: None
    loaded = []
    for short in ('stage_queue_trace', 'stage_study', 'small_inventory'):
        name = 'vllm_omni.profiling.' + short
        saved[name] = sys.modules.get(name)
        spec = importlib.util.spec_from_file_location(name, ROOT / 'vllm_omni/profiling' / (short + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(sys.modules['vllm_omni.profiling'], short, module)
        loaded.append(module)
    for name, old in saved.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old
    return loaded

TRACE, STUDY, INVENTORY = load_modules()


def method_ast(path, name):
    tree = ast.parse((ROOT / path).read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def compile_method(method, namespace):
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<extracted-actual-method>', 'exec'), namespace)
    return namespace[method.name]


class StudyContracts(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'VLLM_OMNI_STAGE_STUDY_TRACE': '1', 'STAGE_QUEUE_TRACE_MODE': 'stage_study'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_inventory_enabled_independently_of_hold(self):
        self.assertTrue(INVENTORY.small_inventory_enabled())
        self.assertNotIn('VLLM_OMNI_SMALL_HOLD_PROBE', os.environ)
        rows = []
        with patch.object(INVENTORY, 'emit_stage_event', lambda event, **kw: rows.append((event, kw))):
            INVENTORY.emit_inventory_event('small_hold_chunk_received', stage_id=2, request_id='r', fields={'new_frames': 3, 'left_context_frames': 25})
        self.assertEqual(rows[0][0], 'stage_study_chunk_received')
        self.assertEqual(rows[0][1]['new_frames'], 3)
        with patch.dict(os.environ, {'VLLM_OMNI_STAGE_STUDY_TRACE': '0'}):
            self.assertFalse(INVENTORY.small_inventory_enabled())

    def test_trace_filter_preserves_inventory_failure_recovery(self):
        for name in ('stage_study_kv_snapshot', 'stage_study_text_conditioning_selected', 'kv_allocation_failed', 'request_preempted', 'stage_recovery_catchup'):
            self.assertTrue(TRACE._record_enabled('stage_events.jsonl', {'event': name}), name)
        self.assertTrue(TRACE._record_enabled('iteration_events.jsonl', {'event': 'iteration'}))
        self.assertFalse(TRACE._record_enabled('stage_events.jsonl', {'event': 'forward_span'}))

    def test_ready_excludes_prompt_and_one_sampled_token_embedding_lag(self):
        prefill = {'ids': {'all': [1, 2, 3, 4, 5], 'prompt': [1, 2, 3]}, 'embed': {'prefill': object()}}
        self.assertEqual(STUDY.text_ready_fields(prefill)['conditioning_ready_index_exclusive'], 1)
        decode = {'ids': {'output': [4, 5, 6, 7]}, 'embed': {'decode': object()}}
        self.assertEqual(STUDY.text_ready_fields(decode)['conditioning_ready_index_exclusive'], 3)

    def test_actual_decode_branch_counts_real_embedding_but_not_eos_pad(self):
        fn = compile_method(method_ast('vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py', '_thinker_decode_to_talker_decode'), {'stage_study': STUDY, 'TalkerConditioningUnavailable': RuntimeError})
        class Tensor:
            device = 'cpu'
            shape = (4, 8)
            def to(self, device): return self
            def __getitem__(self, index): return self
        tensor = Tensor()
        model = types.SimpleNamespace(tts_pad_embed=tensor, tts_eos_embed=tensor, talker=types.SimpleNamespace(text_projection=lambda x:x))
        payload = {'request_id':'physical', 'meta':{'num_processed_tokens':1, 'stage_study_identity':{'logical_request_id':'logical','execution_id':'attempt-1','pipeline_attempt_id':1,'stage_attempt_id':0}}, 'ids':{'output':[10,11,12,13]}, 'embed':{'cached_decode':tensor}}
        rows=[]
        with patch.object(STUDY,'emit_stage_event',lambda e,**kw:rows.append(kw)):
            fn(model,payload,'cpu',{})
            payload['meta']['num_processed_tokens']=3
            with self.assertRaises(RuntimeError):
                fn(model,payload,'cpu',{})  # Nonterminal input gap cannot become EOS.
            payload['meta']['finished']=True
            updates={};fn(model,payload,'cpu',updates)
            payload['meta'].update(updates['meta']);fn(model,payload,'cpu',{})
        self.assertEqual([r['conditioning_kind'] for r in rows], ['output_embedding','eos','pad'])
        self.assertEqual([r['selected_output_embeddings'] for r in rows], [1,0,0])
        self.assertEqual(rows[0]['logical_request_id'],'logical')
        self.assertEqual(rows[0]['execution_id'],'attempt-1')
        self.assertEqual(rows[0]['start_index'],1)

    def test_preempt_snapshot_is_before_free_and_original_called_once(self):
        pool=types.SimpleNamespace(num_gpu_blocks=10, get_num_free_blocks=lambda:4)
        req=types.SimpleNamespace(request_id='r',num_prompt_tokens=8,num_computed_tokens=9,output_token_ids=[3,4],additional_information={},status=types.SimpleNamespace(name='RUNNING'))
        manager=types.SimpleNamespace(block_pool=pool,usage=0.6,get_block_ids=lambda rid:([1,2,3],))
        class Base:
            calls=0
            def _preempt_request(self, request, timestamp):
                self.calls+=1
                self.kv_cache_manager.block_pool.get_num_free_blocks=lambda:7
                request.num_computed_tokens=0
        fn=method_ast('vllm_omni/core/sched/omni_scheduler_mixin.py','_preempt_request')
        cls=ast.ClassDef(name='Mixin',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[fn],decorator_list=[])
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
        ns={'Base':Base,'stage_study':STUDY,'deserialize_additional_information':lambda x:x}
        exec(compile(ast.fix_missing_locations(module),'<actual-mixin-hook>','exec'),ns)
        scheduler=ns['Mixin']();scheduler.kv_cache_manager=manager;scheduler.kv_cache_config=types.SimpleNamespace(kv_cache_groups=[])
        scheduler._omni_stage_id_for_trace=lambda:0
        scheduler._stage_study_snapshot=lambda **kw:None
        rows=[]
        with patch.object(STUDY,'emit_stage_event',lambda e,**kw:rows.append(kw)):
            scheduler._preempt_request(req,1.0)
        self.assertEqual(scheduler.calls,1)
        self.assertEqual(rows[0]['kv_free_blocks'],4)
        self.assertEqual(rows[0]['num_computed_tokens'],9)
        self.assertEqual(rows[0]['allocated_blocks_per_group'],[3])
        self.assertEqual(req.num_computed_tokens,0)

    def test_snapshot_includes_chunk_waiting_map_request_once_without_mutation(self):
        fn = compile_method(method_ast('vllm_omni/core/sched/omni_scheduler_mixin.py', '_stage_study_snapshot'),
                            {'stage_study': STUDY, 'time': types.SimpleNamespace(monotonic=lambda:100.0),
                             'os': os, 'deserialize_additional_information': lambda x:x})
        def request(rid, status):
            return types.SimpleNamespace(request_id=rid, num_prompt_tokens=32, num_computed_tokens=33,
                output_token_ids=[1], additional_information={}, status=types.SimpleNamespace(name=status))
        hidden=request('hidden','WAITING_FOR_CHUNK'); live=request('live','RUNNING'); residual=request('residual','WAITING_FOR_CHUNK')
        adapter=types.SimpleNamespace(waiting_for_chunk_running_requests=[hidden], waiting_for_chunk_waiting_requests=[])
        scheduler=types.SimpleNamespace(requests={'hidden':hidden,'live':live},running=[live,residual],waiting=[live],
            chunk_transfer_adapter=adapter,_omni_stage_id_for_trace=lambda:1,
            kv_cache_manager=types.SimpleNamespace(block_pool=types.SimpleNamespace(num_gpu_blocks=10,get_num_free_blocks=lambda:4),
                                                  get_block_ids=lambda rid:([1,2],)),
            kv_cache_config=types.SimpleNamespace(kv_cache_groups=[]))
        rows=[]
        with patch.object(STUDY,'emit_stage_event',lambda e,**kw:rows.append(kw)):
            fn(scheduler,point='before_schedule')
            fn(scheduler,point='before_schedule')  # Same monotonic time: 50 ms throttle remains active.
        self.assertEqual(len(rows),1)
        observed={r['request_id']:r for r in rows[0]['requests']}
        self.assertEqual(len(observed),3)
        self.assertEqual(observed['hidden']['queue_membership'],['request_map','adapter_waiting_for_chunk_running'])
        self.assertTrue(observed['hidden']['waiting_for_chunk'])
        self.assertEqual(observed['hidden']['allocated_blocks_per_group'],[2])
        self.assertEqual(observed['live']['queue_membership'],['request_map','running','waiting'])
        self.assertEqual(observed['residual']['queue_membership'],['running'])
        self.assertEqual(rows[0]['request_snapshot_scope'],'request_map_union_scheduler_queues_v2')
        self.assertEqual(scheduler.running,[live,residual])
        self.assertEqual(adapter.waiting_for_chunk_running_requests,[hidden])
        self.assertEqual(hidden.num_computed_tokens,33)

    def test_missing_kv_is_unknown_not_zero_usage(self):
        counts=STUDY.kv_counts(types.SimpleNamespace())
        self.assertIsNone(counts['kv_free_blocks'])
        self.assertIsNone(counts['kv_used_blocks'])

    def test_gpu_counters_do_not_call_synchronize_or_reset(self):
        cuda=types.SimpleNamespace(memory_allocated=lambda d:10,memory_reserved=lambda d:20,max_memory_allocated=lambda d:30,max_memory_reserved=lambda d:40)
        runner=types.SimpleNamespace(device='cpu-test',model_config=types.SimpleNamespace(stage_id=0),input_batch=types.SimpleNamespace(req_ids=['r']))
        rows=[]
        with patch.object(STUDY,'emit_stage_event',lambda e,**kw:rows.append(kw)):
            STUDY.gpu_memory(runner,types.SimpleNamespace(cuda=cuda),point='after_encoder',force=True,encoder_request_ids=['r'])
        self.assertEqual(rows[0]['torch_lifetime_peak_allocated_bytes'],30)
        self.assertEqual(rows[0]['encoder_request_ids'],['r'])
        self.assertEqual(rows[0]['request_ids_scope'],'entire_input_batch')
        self.assertIn('not_isolated_encoder',rows[0]['memory_scope'])


if __name__=='__main__':
    unittest.main(verbosity=2)
