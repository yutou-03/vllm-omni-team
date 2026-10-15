from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav


class LengthSensitiveDecoder(torch.nn.Module):
    _chunked_decode_exact_lengths = Qwen3OmniMoeCode2Wav._chunked_decode_exact_lengths

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_quantizers=2)
        self.total_upsample = 8
        self._cudagraph_enabled = False
        self._cudagraph_wrapper = None
        self.calls = []

    def forward(self, codes):
        self.calls.append(tuple(codes.shape))
        # Temporal padding changes the prefix as well as its length.
        length = codes.shape[-1]
        ids = codes[:, 0, 0].reshape(-1, 1, 1)
        return (ids * 100 + length).expand(-1, 1, length * 8 - 3).clone()


def test_mixed_lengths_keep_native_outputs_order_and_fixed_batch_one():
    model = LengthSensitiveDecoder()
    codes = torch.zeros(3, 2, 5, dtype=torch.long)
    codes[:, :, 0] = torch.tensor([1, 2, 3])[:, None]
    actual = model._chunked_decode_exact_lengths(codes, [1, 0, 2], [6, 10, 6])
    assert model.calls == [(1, 2, 3), (1, 2, 5), (1, 2, 3)]
    for i, (length, left) in enumerate(zip([3, 5, 3], [1, 0, 2])):
        expected = model(codes[i:i+1, :, :length])[0, :, left*8:]
        assert torch.equal(actual[i], expected)


def test_only_exact_graph_lengths_are_used():
    model = LengthSensitiveDecoder()
    model._cudagraph_enabled = True
    model._cudagraph_wrapper = SimpleNamespace(graphs={5: object()}, decode=Mock(side_effect=model.forward))
    codes = torch.zeros(2, 2, 5, dtype=torch.long)
    model._chunked_decode_exact_lengths(codes, [0, 0], [6, 10])
    assert model._cudagraph_wrapper.decode.call_count == 1
    assert model._cudagraph_wrapper.decode.call_args.args[0].shape[-1] == 5


@pytest.mark.parametrize('counts,left', [([5], [0]), ([12], [0]), ([0], [0]), ([6], [4]), ([6, 6], [0])])
def test_invalid_metadata_fails_before_decoding(counts, left):
    model = LengthSensitiveDecoder()
    with pytest.raises(ValueError):
        model._chunked_decode_exact_lengths(torch.zeros(1, 2, 5), left, counts)
    assert not model.calls
