# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe Code2Wav model."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from vllm import envs
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
)
from vllm.config import VllmConfig  # type: ignore
from vllm.logger import init_logger  # type: ignore
from vllm.model_executor.models.utils import (  # type: ignore
    AutoWeightsLoader,
    WeightsMapper,
)

from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta
from vllm_omni.profiling.nvtx import nvtx_range

logger = init_logger(__name__)


class Qwen3OmniMoeCode2Wav(nn.Module):
    """
    Qwen3 Omni MoE Code2Wav - Converts num_quantizers-layer RVQ codec codes to audio waveform.

    Architecture:
    1. Code Embedding: Embed and average num_quantizers RVQ layers
    2. Pre-Transformer: Add temporal context via sliding-window attention
    3. Upsampling: Progressive upsampling with ConvNeXt blocks
    4. Decoder: Multi-stage upsampling + residual units → waveform

    Input: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
    Output: [batch, 1, waveform_len] - Audio waveform [-1, 1]

    Total upsampling factor: ~1280x
    Example: 100 codec frames → 128,000 audio samples (8 seconds at 16kHz)
    """

    input_modalities = "audio"

    # Weight mapper
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "code2wav.pre_transformer.": "pre_transformer.",
            "code2wav.code_embedding.": "code_embedding.",
            "code2wav.upsample.": "upsample.",
            "code2wav.decoder.": "decoder.",
            "code2wav.": "",
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.config: Qwen3OmniMoeCode2WavConfig = vllm_config.model_config.hf_config

        # Calculate total upsampling factor
        self.total_upsample = np.prod(self.config.upsample_rates + self.config.upsampling_ratios)

        # Pre-transformer
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(self.config)

        # Code embedding: Single embedding table for all RVQ layers
        self.code_embedding = nn.Embedding(
            self.config.codebook_size * self.config.num_quantizers, self.config.hidden_size
        )

        # Offset for each RVQ layer (layer 0: 0-1023, layer 1: 1024-2047, etc.)
        self.register_buffer(
            "code_offset",
            torch.arange(self.config.num_quantizers).view(1, -1, 1) * self.config.codebook_size,
            persistent=False,
        )

        # Upsampling blocks (e.g., 2x, 2x)
        upsample = []
        for factor in self.config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(
                            self.config.hidden_size, self.config.hidden_size, factor, factor
                        ),
                        Qwen3OmniMoeConvNeXtBlock(self.config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        # Decoder: Initial projection + progressive upsampling blocks
        decoder = [Qwen3OmniMoeCausalConvNet(self.config.hidden_size, self.config.decoder_dim, kernel_size=7)]

        # Add decoder blocks (each upsamples and reduces channels)
        for i in range(len(self.config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(self.config, i))

        # Final projection to waveform
        output_dim = self.config.decoder_dim // 2 ** len(self.config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, kernel_size=7),
        ]
        self.decoder = nn.ModuleList(decoder)

        # CUDA Graph support — reuses CUDAGraphDecoderWrapper from Qwen3-TTS
        self._cudagraph_enabled = False
        self._cudagraph_wrapper = None

    def precompute_snake_caches(self):
        """Precompute exp(alpha) and 1/(exp(beta)+eps) for all SnakeBeta modules."""
        count = 0
        for module in self.modules():
            if isinstance(module, SnakeBeta):
                module.precompute_exp_cache()
                count += 1
        if count > 0:
            logger.info("Precomputed exp caches for %d SnakeBeta activations", count)

    def enable_cudagraph(
        self,
        device: torch.device | None = None,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ):
        """Enable CUDA graph acceleration (same pattern as Qwen3-TTS Code2Wav)."""
        from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import (
            CUDAGraphDecoderWrapper,
        )

        if device is None:
            device = next(self.parameters()).device
        if device.type != "cuda":
            logger.warning("Cannot enable CUDA Graph: not on CUDA device (got %s)", device)
            return

        wrapper = CUDAGraphDecoderWrapper(
            decoder=self,
            num_quantizers=self.config.num_quantizers,
            enabled=True,
        )
        try:
            wrapper.warmup(
                device,
                dtype=torch.long,
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
            )
        except Exception:
            self._cudagraph_wrapper = None
            self._cudagraph_enabled = False
            raise
        self._cudagraph_wrapper = wrapper
        self._cudagraph_enabled = True
        logger.info(
            "CUDA Graph enabled for Code2Wav: num_quantizers=%d, sizes=%s",
            self.config.num_quantizers,
            self._cudagraph_wrapper.capture_sizes,
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert num_quantizers-layer RVQ codes to audio waveform.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codec codes

        Returns:
            waveform: [batch, 1, waveform_len] - Audio waveform clipped to [-1, 1]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layers of codes, got {codes.shape[1]}")

        shape_suffix = f"b={codes.shape[0]}:q={codes.shape[1]}:t={codes.shape[-1]}"

        # Stage 1: Code Embedding
        # Add offset to separate layer vocabularies, then embed and average
        with nvtx_range(f"code2wav:code_embedding:{shape_suffix}", color="magenta"):
            hidden = self.code_embedding(codes + self.code_offset)
            # Batch-invariant mean accumulates and returns FP32. Restore the
            # embedding dtype expected by the pre-transformer's linear layers,
            # matching native mean's output dtype without changing accumulation.
            hidden = hidden.mean(1).to(dtype=hidden.dtype)
        # Shape: [batch, seq_len, hidden_size]

        # Stage 2: Pre-Transformer (add temporal context)
        with nvtx_range(f"code2wav:pre_transformer:{shape_suffix}", color="blue"):
            hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        # Shape: [batch, seq_len, hidden_size]

        # Stage 3: Upsampling
        with nvtx_range(f"code2wav:upsample:{shape_suffix}", color="orange"):
            hidden = hidden.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
            for block_group_idx, blocks in enumerate(self.upsample):
                with nvtx_range(f"code2wav:upsample_group:{block_group_idx}", color="orange"):
                    for block_idx, block in enumerate(blocks):
                        with nvtx_range(f"code2wav:upsample_block:{block_group_idx}.{block_idx}", color="orange"):
                            hidden = block(hidden)
        # Shape: [batch, hidden_size, seq_len * upsample_factor]

        # Stage 4: Decoder (progressive upsampling to waveform)
        with nvtx_range(f"code2wav:decoder:{shape_suffix}", color="red"):
            wav = hidden
            for block_idx, block in enumerate(self.decoder):
                with nvtx_range(f"code2wav:decoder_block:{block_idx}", color="red"):
                    wav = block(wav)
        # Shape: [batch, 1, waveform_len]

        # Clamp to valid audio range
        with nvtx_range(f"code2wav:clamp:{shape_suffix}", color="green"):
            return wav.clamp(min=-1.0, max=1.0)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.
        When CUDA graphs are enabled, delegates chunk-level decoding to the
        CUDAGraphDecoderWrapper for reduced kernel launch overhead.

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            chunk_size: Number of codec frames per chunk
            left_context_size: Number of overlapping frames for context
            seq_token_counts: Token count for each request in batch

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        # Use CUDA graph wrapper for chunk-level decode when available
        if self._cudagraph_enabled and self._cudagraph_wrapper is not None:
            with nvtx_range(
                f"code2wav:chunked_decode_cudagraph:t={codes.shape[-1]}:chunk={chunk_size}:ctx={left_context_size}",
                color="red",
            ):
                batch_wav = self._cudagraph_wrapper.chunked_decode_with_cudagraph(
                    codes,
                    chunk_size,
                    left_context_size,
                )
        else:
            wavs = []
            start_index = 0

            while start_index < codes.shape[-1]:
                end_index = min(start_index + chunk_size, codes.shape[-1])
                context_size = left_context_size if start_index >= left_context_size else start_index

                # Extract chunk with left context
                with nvtx_range(
                    f"code2wav:chunk_slice:start={start_index}:end={end_index}:ctx={context_size}",
                    color="magenta",
                ):
                    codes_chunk = codes[..., start_index - context_size : end_index]

                # Decode chunk
                with nvtx_range(
                    f"code2wav:chunk_forward:start={start_index}:end={end_index}:ctx={context_size}",
                    color="red",
                ):
                    wav_chunk = self(codes_chunk)

                # Remove context from output (context_size * total_upsample samples)
                with nvtx_range(
                    f"code2wav:chunk_trim:start={start_index}:ctx={context_size}",
                    color="green",
                ):
                    wavs.append(wav_chunk[..., context_size * self.total_upsample :])

                start_index = end_index

            with nvtx_range(f"code2wav:chunk_concat:n={len(wavs)}", color="green"):
                batch_wav = torch.cat(wavs, dim=-1)

        if seq_token_counts is not None:
            code_seq_lens = [seq_len // self.config.num_quantizers for seq_len in seq_token_counts]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        result = []
        with nvtx_range(f"code2wav:batch_trim:n={len(code_seq_lens)}", color="green"):
            for idx, code_seq_len in enumerate(code_seq_lens):
                wav_chunk = batch_wav[idx, :, : code_seq_len * self.total_upsample]
                result.append(wav_chunk)
        return result

    def chunked_decode_streaming(
        self,
        codes: torch.Tensor,
        left_context_size: list[int] | None = None,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Decode long sequences in chunks to avoid OOM.

        Uses overlapping chunks with left context to avoid boundary artifacts.

        No longer need chunk size here, which is different from chunked_decode

        Args:
            codes: [batch, num_quantizers, seq_len] - num_quantizers-layer RVQ codes
            left_context_size: Number of overlapping frames for context
            seq_token_counts: Token count for each request in batch

        Returns:
            list[torch.Tensor]: Complete waveform decoded from the input
                codes. For ``batch_size == 1``, this is a list containing a
                single tensor with shape ``[1, waveform_len]``.
        """
        if not (left_context_size and seq_token_counts and len(left_context_size) == len(seq_token_counts)):
            logger.warning(
                "chunked_decode_streaming: missing/invalid left_context_size or seq_token_counts; "
                "defaulting to left_context_size=zeros(len=codes.shape[0])."
            )
            left_context_size = [0] * codes.shape[0]
        if envs.VLLM_BATCH_INVARIANT and seq_token_counts is not None:
            return self._chunked_decode_exact_lengths(codes, left_context_size, seq_token_counts)
        # Decode chunk
        wavs = []
        if self._cudagraph_enabled and self._cudagraph_wrapper is not None:
            with nvtx_range(f"code2wav:streaming_cudagraph_decode:t={codes.shape[-1]}", color="red"):
                batch_wav = self._cudagraph_wrapper.decode(codes)
        else:
            with nvtx_range(f"code2wav:streaming_forward:t={codes.shape[-1]}", color="red"):
                batch_wav = self(codes)
        if seq_token_counts is not None:
            code_seq_lens = [n // self.config.num_quantizers for n in seq_token_counts]
        else:
            # Fallback: assume all batch elements share the same sequence length.
            code_seq_lens = [codes.shape[-1]] * codes.shape[0]
        with nvtx_range(f"code2wav:streaming_batch_trim:n={len(code_seq_lens)}", color="green"):
            for idx, code_seq_len in enumerate(code_seq_lens):
                # Remove context from output (left_context_size * total_upsample samples)
                wav_chunk = batch_wav[
                    idx, :, left_context_size[idx] * self.total_upsample : code_seq_len * self.total_upsample
                ]
                wavs.append(wav_chunk)
        return wavs

    def _chunked_decode_exact_lengths(
        self, codes: torch.Tensor, left_context_size: list[int], seq_token_counts: list[int]
    ) -> list[torch.Tensor]:
        """Keep temporal padding out of the batch-invariant streaming path.

        Decoding padded RVQ frames changes both the native waveform length and
        its valid prefix. Cropping the padded waveform afterwards cannot undo
        that change. Even equal-length batches can change BF16 waveforms for
        some code values, so this opt-in diagnostic path also fixes batch=1.
        """
        if len(seq_token_counts) != codes.shape[0] or len(left_context_size) != codes.shape[0]:
            raise ValueError("Streaming metadata must contain one entry per request")
        lengths = []
        for index, count in enumerate(seq_token_counts):
            length, remainder = divmod(count, self.config.num_quantizers)
            if remainder or not 0 < length <= codes.shape[-1]:
                raise ValueError("Invalid per-request streaming codec count")
            if not 0 <= left_context_size[index] <= length:
                raise ValueError("Invalid per-request streaming left context")
            lengths.append(length)
        outputs = []
        for index, length in enumerate(lengths):
            batch = codes[index : index + 1, :, :length].contiguous()
            wrapper = self._cudagraph_wrapper
            # The graph wrapper otherwise pads to its next capture bucket.
            # Exact captures are safe; unmatched lengths use native forward.
            if self._cudagraph_enabled and wrapper is not None and length in wrapper.graphs:
                waveforms = wrapper.decode(batch)
            else:
                waveforms = self(batch)
            outputs.append(waveforms[0, :, left_context_size[index] * self.total_upsample :])
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from HuggingFace checkpoint."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "talker."],  # Already loaded above
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            logger.error("Error logging model load summary")

        return loaded
