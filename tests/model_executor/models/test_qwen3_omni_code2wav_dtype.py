# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _LinearTransformer(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False, dtype=dtype)
        self.seen_dtype = None

    def forward(self, *, inputs_embeds):
        self.seen_dtype = inputs_embeds.dtype
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("promoted_mean", [False, True])
def test_codebook_mean_preserves_transformer_dtype_and_values(dtype, promoted_mean):
    model = Qwen3OmniMoeCode2Wav.__new__(Qwen3OmniMoeCode2Wav)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_quantizers=4)
    model.code_embedding = nn.Embedding(64, 8, dtype=dtype)
    model.register_buffer("code_offset", torch.arange(4).view(1, -1, 1) * 16)
    model.pre_transformer = _LinearTransformer(dtype)
    model.upsample = nn.ModuleList()
    model.decoder = nn.ModuleList()
    codes = torch.arange(40).reshape(2, 4, 5) % 16
    native_mean = torch.Tensor.mean
    with torch.inference_mode():
        embedding = model.code_embedding(codes + model.code_offset)
        expected = model.pre_transformer(inputs_embeds=native_mean(embedding, 1)).last_hidden_state
        expected = expected.permute(0, 2, 1).clamp(-1, 1)

        def mean_with_fp32_result(tensor, *args, **kwargs):
            return native_mean(tensor.float(), *args, **kwargs)

        if promoted_mean:
            with patch.object(torch.Tensor, "mean", mean_with_fp32_result):
                actual = model(codes)
        else:
            actual = model(codes)
    assert model.pre_transformer.seen_dtype == dtype
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
