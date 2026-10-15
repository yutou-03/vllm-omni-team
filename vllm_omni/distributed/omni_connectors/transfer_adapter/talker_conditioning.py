# SPDX-License-Identifier: Apache-2.0
"""Scheduler-visible readiness for Qwen3-Omni Thinker-to-Talker chunks."""

from collections.abc import Mapping
from typing import Any


def is_qwen3_omni_async_talker(model_config: Any) -> bool:
    """Identify the Talker from fields carried into the engine model config.

    The input processor belongs to frontend StageConfig and is absent from
    the scheduler's OmniModelConfig. The next-stage processor is propagated.
    """
    return (
        bool(getattr(model_config, "async_chunk", False))
        and getattr(model_config, "model_arch", None)
        == "Qwen3OmniMoeForConditionalGeneration"
        and getattr(model_config, "model_stage", None) == "talker"
        and str(getattr(model_config, "custom_process_next_stage_input_func", ""))
        .endswith("qwen3_omni.talker2code2wav_async_chunk")
    )


def qwen3_talker_conditioning_ready(
    payload: Mapping[str, Any] | None,
    generated_codec_tokens: int,
    *,
    upstream_finished: bool = False,
) -> bool:
    """True when the next Talker step has its Thinker text embedding.

    The Thinker forwards embeddings one sampled token behind its output IDs.
    Talker prefill consumes embedding index 0; after generating N codec tokens,
    its next decode consumes embedding index N. Only an explicit upstream
    terminal marker permits Talker to substitute EOS/PAD for missing text.
    """
    if upstream_finished:
        return True
    if not isinstance(payload, Mapping):
        return False
    ids = payload.get("ids")
    if not isinstance(ids, Mapping):
        return False
    output_ids = ids.get("output")
    if output_ids is not None:
        total = len(output_ids)
    else:
        all_ids, prompt_ids = ids.get("all"), ids.get("prompt")
        if all_ids is None or prompt_ids is None:
            return False
        total = max(len(all_ids) - len(prompt_ids), 0)
    ready_exclusive = max(total - 1, 0)
    required_exclusive = max(int(generated_codec_tokens) + 1, 1)
    return ready_exclusive >= required_exclusive
