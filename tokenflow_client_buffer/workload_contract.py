"""Fixed-size audio benchmark inputs; never used by production admission.

Qwen3-Omni's first sampled Talker token follows prefill, whose codec output is
an excluded zero placeholder. N sampled tokens yield N-1 delivered RVQ frames.
Revalidate this correspondence when changing model implementations.
"""
import copy


def fixed_audio_request(row, *, thinker_tokens, codec_frames, codec_eos_id):
    for value in (thinker_tokens, codec_frames):
        if type(value) is not int or value <= 0:
            raise ValueError("Output targets must be positive integers")
    result = copy.deepcopy(row)
    payload = result['payload']
    if payload.get('modalities') != ['audio']:
        raise ValueError('Expected pure audio request')
    stages = payload.get('sampling_params_list')
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError('Explicit sampling parameters for all three stages required')
    payload.update(max_tokens=thinker_tokens, min_tokens=thinker_tokens, ignore_eos=True)
    for params, target in zip(stages[:2], (thinker_tokens, codec_frames + 1)):
        params.update(max_tokens=target, min_tokens=target, ignore_eos=True)
        # min_tokens suppresses stop IDs as well as EOS. A stop string could
        # otherwise terminate a supposedly fixed-work request early.
        params.pop('stop', None)
    stages[1]['stop_token_ids'] = sorted(set(stages[1].get('stop_token_ids', [])) | {codec_eos_id})
    payload.pop('stop', None)
    return result


def expected_chunks(frames, chunk_frames, context_frames):
    """Unique output frames and repeated decoder context, excluding padding."""
    if frames <= 0 or chunk_frames <= 0 or context_frames < 0:
        raise ValueError('Invalid codec geometry')
    return [dict(new_frames=min(chunk_frames, frames-start),
                 left_context_frames=min(start, context_frames),
                 codec_payload_frames=min(chunk_frames, frames-start)+min(start, context_frames))
            for start in range(0, frames, chunk_frames)]
