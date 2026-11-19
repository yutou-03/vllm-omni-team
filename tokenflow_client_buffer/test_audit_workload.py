import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
from .audit_workload import summarize_events, wire_counts


class AuditWorkloadTests(unittest.TestCase):
    def test_real_wav_counts_and_requires_done(self):
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as w:
            w.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            w.writeframes(b'\x00\x00'*37)
        event = dict(modality='audio', choices=[dict(delta=dict(content=base64.b64encode(stream.getvalue()).decode()))])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'stream.sse'
            body = 'data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n'
            path.write_text(body)
            self.assertEqual(wire_counts(path), [dict(samples=37, rate=24000, channels=1)])
            path.write_text(body.replace('data: [DONE]\n\n', ''))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                wire_counts(path)

    def test_contract_maps_earlier_events_without_cross_request_join(self):
        es = [dict(event='stage_study_codec_frame_produced', request_id='physical', codec_total_frames=1),
              dict(event='client_buffer_contract', request_id='physical', logical_request_id='logical', source_request_id='wanted'),
              dict(event='stage_study_chunk_conversion_completed', request_id='logical'),
              dict(event='stage_study_chunk_conversion_completed', request_id='unrelated')]
        result = summarize_events(es, {'wanted'})
        self.assertEqual(len(result['wanted']), 3)
        self.assertEqual(set(result), {'wanted'})


if __name__ == '__main__':
    unittest.main()
