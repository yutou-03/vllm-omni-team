import unittest
from .workload_contract import expected_chunks, fixed_audio_request


class WorkloadContractTest(unittest.TestCase):
    def test_fixed_lengths_preserve_request_and_decoder(self):
        row = dict(arrival_s=3, payload=dict(modalities=['audio'], messages=['same'],
                   sampling_params_list=[{'stop': ['early']}, {'stop_token_ids': [7]}, {'max_tokens': 65536}]))
        new = fixed_audio_request(row, thinker_tokens=96, codec_frames=151, codec_eos_id=2150)
        self.assertEqual(row['payload']['sampling_params_list'][0], {'stop': ['early']})
        self.assertEqual(new['payload']['messages'], row['payload']['messages'])
        self.assertEqual(new['payload']['sampling_params_list'][2], row['payload']['sampling_params_list'][2])
        for p, n in zip(new['payload']['sampling_params_list'], [96, 152]):
            self.assertEqual((p['min_tokens'], p['max_tokens'], p['ignore_eos']), (n, n, True))
            self.assertNotIn('stop', p)
        self.assertEqual(new['payload']['sampling_params_list'][1]['stop_token_ids'], [7, 2150])

    def test_tail_and_context_counted_separately(self):
        chunks = expected_chunks(151, 25, 25)
        self.assertEqual(sum(c['new_frames'] for c in chunks), 151)
        self.assertEqual(sum(c['codec_payload_frames'] for c in chunks), 301)
        self.assertEqual(chunks[-1], dict(new_frames=1, left_context_frames=25, codec_payload_frames=26))
        self.assertEqual(len(expected_chunks(100, 25, 25)), 4)

    def test_reject_zero_and_non_integer_targets(self):
        for n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                fixed_audio_request({}, thinker_tokens=n, codec_frames=10, codec_eos_id=2150)


if __name__ == '__main__':
    unittest.main()
