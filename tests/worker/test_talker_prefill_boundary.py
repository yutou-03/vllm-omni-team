"""Initial Talker prompt chunks must finish before codec MTP decode."""
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_model(prompt_len=7):
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.talker = SimpleNamespace(num_code_groups=2, text_projection=lambda x: x)
    model.default_tts_text_spk_type = 'test'
    model._module_device = lambda _: torch.device('cpu')
    model._get_text_spk_token_id = lambda _: 0
    ids = torch.arange(prompt_len, dtype=torch.long)
    embeds = torch.arange(prompt_len * 4, dtype=torch.float32).reshape(prompt_len, 4)
    model._thinker_to_talker_prefill = lambda **_: (ids, embeds, torch.empty(0, 4), None)
    calls = []

    def decode(input_ids, input_embeds, updates, payload):
        calls.append(payload['meta']['num_processed_tokens'])
        return torch.zeros(1, 4), torch.zeros(1, 4), updates

    model.talker_preprocess_decode = decode
    payload = {
        'meta': {},
        'hidden_states': {'output': torch.zeros(1, 4)},
        'ids': {'all': ids, 'prompt': ids},
        'embed': {'prefill': embeds, 'tts_bos': torch.zeros(1, 4),
                  'tts_eos': torch.zeros(1, 4), 'tts_pad': torch.zeros(1, 4)},
    }
    return model, payload, calls, ids, embeds


def apply_step(model, payload, tokens, **kwargs):
    result = model.talker_preprocess(torch.zeros(tokens, dtype=torch.long), torch.zeros(tokens, 4), **payload, **kwargs)
    for key, value in result[2].items():
        if isinstance(value, dict):
            payload.setdefault(key, {}).update(value)
    return result


def test_single_token_final_prefill_chunk_hands_off_only_after_bootstrap():
    model, payload, calls, ids, embeds = make_model()
    first = apply_step(model, payload, 6)
    assert 'prefill_consumed_text_tokens' not in payload['meta']
    tail = apply_step(model, payload, 1)
    assert torch.equal(torch.cat([first[0], tail[0]]), ids)
    assert torch.equal(torch.cat([first[1], tail[1]]), embeds)
    assert 'mtp_inputs' not in tail[2]
    assert payload['meta']['prefill_consumed_text_tokens'] == 1
    assert calls == []
    decoded = apply_step(model, payload, 1)
    assert 'mtp_inputs' in decoded[2]
    assert calls == [1]
    assert payload['meta']['num_processed_tokens'] == 2


def test_first_and_subsequent_one_token_prefill_chunks():
    model, payload, calls, ids, embeds = make_model()
    result_ids = []
    result_embeds = []
    for position in range(len(ids)):
        phase = model.talker_is_prefill(1, payload, num_computed_tokens=position)
        assert phase
        out_ids, out_embeds, updates = apply_step(model, payload, 1, is_prefill=phase)
        assert 'mtp_inputs' not in updates
        result_ids.append(out_ids)
        result_embeds.append(out_embeds)
    assert torch.equal(torch.cat(result_ids), ids)
    assert torch.equal(torch.cat(result_embeds), embeds)
    assert calls == []
    assert not model.talker_is_prefill(1, payload, num_computed_tokens=len(ids))
    apply_step(model, payload, 1, is_prefill=False)
    assert calls == [1]


def test_decode_without_valid_prefill_handoff_still_fails():
    model, payload, calls, _, _ = make_model()
    payload['meta']['num_processed_tokens'] = 7
    with pytest.raises(RuntimeError, match='Missing prefill_consumed_text_tokens'):
        apply_step(model, payload, 1, is_prefill=False)
    assert calls == []


def test_decode_phase_is_not_reset_by_streaming_prompt_growth():
    model, payload, calls, _, _ = make_model()
    apply_step(model, payload, 7)
    assert not model.talker_is_prefill(1, payload, num_computed_tokens=7)
    apply_step(model, payload, 1, is_prefill=False)
    assert not model.talker_is_prefill(1, payload, num_computed_tokens=8)
    assert calls == [1]
