"""CPU regression for actual PyTorch pickle amplification; no inference fixtures."""
import importlib.util
from pathlib import Path
import pickle
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[2] / 'vllm_omni/utils/tensor_storage.py'
spec = importlib.util.spec_from_file_location('tensor_storage_under_test', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compact = module.compact_cpu_tensor_storage


class TensorStorageTests(unittest.TestCase):
    def test_contiguous_slice_pickle_does_not_repeat_entire_batch(self):
        batch = torch.arange(1024*32).reshape(1024, 32).to(torch.bfloat16)
        payload = [{'hidden': batch[i:i+1]} for i in range(16)]
        self.assertTrue(payload[1]['hidden'].is_contiguous())
        before = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        after = pickle.dumps(compact(payload), protocol=pickle.HIGHEST_PROTOCOL)
        self.assertGreater(len(before), len(after)*50)
        decoded = pickle.loads(after)
        for i, row in enumerate(decoded):
            self.assertTrue(torch.equal(row['hidden'], batch[i:i+1]))
            self.assertEqual(row['hidden'].dtype, torch.bfloat16)

    def test_first_slice_at_zero_offset_is_also_compacted(self):
        parent = torch.ones(100, 4)
        result = compact(parent[:2])
        self.assertEqual(result.untyped_storage().nbytes(), 2*4*4)
        self.assertTrue(torch.equal(result, parent[:2]))

    def test_owned_compact_tensor_is_not_copied(self):
        tensor = torch.arange(16)
        self.assertIs(compact(tensor), tensor)

    def test_nested_conditioning_shapes_values_and_metadata(self):
        tensor = torch.arange(100).reshape(10, 10)
        payload = {'hidden': tensor[3:6], 'embed': {'prefill': tensor[:3]},
                   'codec': [tensor[:, ::2], 7], 'flags': (True, None), 'id': 'r'}
        result = pickle.loads(pickle.dumps(compact(payload)))
        self.assertTrue(torch.equal(result['hidden'], payload['hidden']))
        self.assertTrue(torch.equal(result['embed']['prefill'], payload['embed']['prefill']))
        self.assertTrue(torch.equal(result['codec'][0], payload['codec'][0]))
        self.assertEqual(result['flags'], (True, None))
        self.assertEqual(result['id'], 'r')

    def test_noncontiguous_transpose_preserves_values(self):
        tensor = torch.arange(30).reshape(5, 6).T
        result = compact(tensor)
        self.assertTrue(torch.equal(result, tensor))
        self.assertTrue(result.is_contiguous())

    def test_empty_slice_retains_no_parent_storage(self):
        tensor = torch.ones(1000)[5:5]
        result = compact(tensor)
        self.assertEqual(result.shape, tensor.shape)
        self.assertEqual(result.untyped_storage().nbytes(), 0)

    def test_repeated_tensor_reference_is_preserved(self):
        tensor = torch.ones(100)[3:6]
        result = compact([tensor, {'same': tensor}])
        self.assertIs(result[0], result[1]['same'])

    def test_source_storage_and_other_request_are_not_mutated(self):
        parent = torch.arange(100)
        result = compact({'a': parent[0:2], 'b': parent[1:3]})
        result['a'][1] = -1
        self.assertEqual(parent[1].item(), 1)
        self.assertEqual(result['b'][0].item(), 1)

    def test_non_cpu_and_non_strided_values_are_untouched(self):
        meta = torch.empty(10, device='meta')
        sparse = torch.eye(3).to_sparse()
        self.assertIs(compact(meta), meta)
        self.assertIs(compact(sparse), sparse)


if __name__ == '__main__':
    unittest.main()
