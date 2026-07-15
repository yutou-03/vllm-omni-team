import base64
from dataclasses import dataclass
import io
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import soundfile as sf
import torch
from vllm.benchmarks.datasets import RandomMultiModalDataset, SampleRequest, process_image, process_video
from vllm.tokenizers import TokenizerLike

from vllm_omni.benchmarks.data_modules.servegen_schema import (
    load_servegen_jsonl,
    mm_item_config,
)

logger = logging.getLogger(__name__)


def process_audio(audio: Any) -> Mapping[str, Any]:
    """
    Process a single audio input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw audio bytes: - Expects a dict with a 'bytes' key
       containing raw audio data.

    2. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the audio URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(audio, dict) and "bytes" in audio:
        audio_bytes = audio["bytes"]
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/mpeg;base64,{audio_base64}"},
        }
    if isinstance(audio, str):
        audio_url = audio if audio.startswith(("http://", "https://", "file://")) else f"file://{audio}"
        return {"type": "audio_url", "audio_url": {"url": audio_url}}

    raise ValueError(
        f"Invalid audio input {audio}. Must be a string of local path/remote url, "
        f"or a dictionary with raw audio bytes in the form of `{{'bytes': raw_audio_bytes}}`."
    )


# -----------------------------------------------------------------------------
# MultiModalDataset Implementation
# -----------------------------------------------------------------------------
class OmniRandomMultiModalDataset(RandomMultiModalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_synthetic_audio(
        self,
        duration: int,  # seconds
        num_channels: int,  # 1:Mono，2:Stereo 5:5.1 surround sound
    ) -> dict[str, Any]:
        """Generate synthetic audio with random values.
        Default use 48000Hz.
        """
        sample_rate = 48000
        num_samples = int(sample_rate * duration)
        audio_data = self._rng.uniform(-0.5, 0.5, (num_samples, num_channels))
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_tensor = torch.FloatTensor(audio_data.T)
        audio_np = audio_tensor.numpy()

        buffer = io.BytesIO()

        sf.write(buffer, audio_np.T, sample_rate, format="wav")

        buffer.seek(0)
        audio_bytes = buffer.read()
        buffer.close()
        return {
            "bytes": audio_bytes,
        }

    def generate_mm_item(
        self,
        mm_item_config: tuple[int, int, int],
    ) -> Mapping[str, Any]:
        """
        Create synthetic images and videos and
        apply process_image/process_video respectively.
        This follows the OpenAI API chat completions
        https://github.com/openai/openai-python
        """

        if self.map_config_to_modality(mm_item_config) == "image":
            return process_image(self.generate_synthetic_image(mm_item_config[1], mm_item_config[0]))
        elif self.map_config_to_modality(mm_item_config) == "video":
            return process_video(self.generate_synthetic_video(mm_item_config[1], mm_item_config[0], mm_item_config[2]))
        elif self.map_config_to_modality(mm_item_config) == "audio":
            return process_audio(self.generate_synthetic_audio(mm_item_config[1], mm_item_config[2]))
        else:
            raise ValueError(f"Invalid multimodal item configuration: {mm_item_config}")

    def generate_synthetic_video(self, width: int, height: int, num_frames: int) -> Any:
        """Generate synthetic video with random values."""
        import imageio

        video_data = self._rng.integers(
            0,
            256,
            (num_frames, height, width, 3),
            dtype=np.uint8,
        )
        buffer = io.BytesIO()
        writer_kwargs = {
            "format": "mp4",
            "fps": 30,
            "codec": "libx264",
            "quality": 7,
            "pixelformat": "yuv420p",
            "macro_block_size": 16,
            "ffmpeg_params": [
                "-preset",
                "medium",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                f"scale={width}:{height}",
            ],
        }

        with imageio.get_writer(buffer, **writer_kwargs) as writer:
            for frame_idx in range(num_frames):
                writer.append_data(video_data[frame_idx])
        buffer.seek(0)
        video_bytes = buffer.read()

        return {
            "bytes": video_bytes,
        }

    def map_config_to_modality(self, config: tuple[int, int, int]) -> str:
        """Map the configuration to the modality."""
        if config[0] == 0:
            return "audio"
        elif config[-1] == 1:
            return "image"
        elif config[-1] > 1:
            return "video"
        else:
            raise ValueError(f"Invalid multimodal item configuration: {config}")


@dataclass
class ServeGenSampleRequest(SampleRequest):
    time_stamp: float = 0.0
    output_modalities: list[str] | None = None
    slo_ms: float | None = None
    request_path: str | None = None
    predicted_stage_ms: dict[str, float] | None = None


class ServeGenDataSet(OmniRandomMultiModalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disable_shuffle = True
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        if not self.dataset_path.endswith(".jsonl"):
            raise NotImplementedError(
                "Only JSONL format is supported for CustomDataset."
            )
        self.data = load_servegen_jsonl(self.dataset_path)

    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int | None = None,
        request_id_prefix: str = "",
        prefix_len: int = 0,
        **kwargs,
    ) -> list[ServeGenSampleRequest]:
        self.num_available_samples = len(self.data)
        if num_requests is None:
            num_requests = len(self.data)
        if num_requests <= 0:
            num_requests = self.num_available_samples
            logger.info(
                "num_requests is set to 0 or negative, "
                "so using all available samples: %d",
                num_requests,
            )
        logger.info("Need Sampling %d requests from the dataset", num_requests)
        sample_requests: list[ServeGenSampleRequest] = []
        prohibited_tokens = list(
            tok_id
            for tok_id, token in tokenizer.added_tokens_decoder.items()
            if token.special
        )
        vocab_size = tokenizer.vocab_size
        all_tokens = np.arange(vocab_size)
        allowed_tokens = np.array(list(set(all_tokens) - set(prohibited_tokens)))
        prefix_token_ids = self.get_prefix(tokenizer, allowed_tokens, prefix_len)
        token_mismatch_total = 0
        seen_request_ids: set[str] = set()
        for i, item in enumerate(self.data[:num_requests]):
            input_len = int(item.get("text_tokens", item.get("input_tokens", 0)))
            prompt, total_input_len, token_mismatch = self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=input_len,
                offset=0,
                index=i,
                allowed_tokens=allowed_tokens,
            )
            token_mismatch_total += token_mismatch
            mm_item_list = [
                self.generate_mm_item(mm_item_config(mm, request_id=item["request_id"]))
                for mm in item["mm_items"]
            ]
            request_id = f"{request_id_prefix}{item['request_id']}"
            if request_id in seen_request_ids:
                raise ValueError(f"Duplicate sampled request_id {request_id!r}")
            seen_request_ids.add(request_id)
            sample_request = ServeGenSampleRequest(
                prompt=prompt,
                prompt_len=total_input_len,
                expected_output_len=item["output_tokens"],
                multi_modal_data=mm_item_list,
                request_id=request_id,
                time_stamp=item["timestamp"],
                output_modalities=item.get("output_modalities"),
                slo_ms=item.get("slo_ms"),
                request_path=item.get("request_path"),
                predicted_stage_ms=item.get("predicted_stage_ms"),
            )
            sample_requests.append(sample_request)
        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                sign,
            )
        return sample_requests
