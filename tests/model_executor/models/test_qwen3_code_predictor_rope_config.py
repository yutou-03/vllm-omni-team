from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.common.qwen3_code_predictor import _RotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("parameters", [
    {"rope_theta": 1_000_000.0},
    {"rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0}},
])
def test_checkpoint_rope_matches_transformers_reference(parameters):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeRotaryEmbedding
    config = SimpleNamespace(hidden_size=1024, num_attention_heads=16, head_dim=128, **parameters)
    reference_config = SimpleNamespace(
        hidden_size=1024, num_attention_heads=16, head_dim=128,
        max_position_embeddings=32768,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
    )
    actual = _RotaryEmbedding(config)
    reference = Qwen3OmniMoeRotaryEmbedding(reference_config)
    torch.testing.assert_close(actual.inv_freq, reference.inv_freq, rtol=0, atol=0)
    positions = torch.tensor([[0, 1, 8, 16]])
    x = torch.zeros(1, 4, 128, dtype=torch.bfloat16)
    for a, b in zip(actual(x, positions), reference(x, positions)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_legacy_default_is_preserved():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=4)
    torch.testing.assert_close(_RotaryEmbedding(config).inv_freq, torch.tensor([1.0, 0.01]))
