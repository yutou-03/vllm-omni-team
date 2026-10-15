# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.common import qwen3_code_predictor as cp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("invariant", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
def test_batch_invariance_retains_operator_dispatch_and_graph_setup(monkeypatch, invariant, graphs):
    config = SimpleNamespace(
        num_code_groups=2, hidden_size=8, num_attention_heads=2, num_key_value_heads=2,
        head_dim=4, num_hidden_layers=1, intermediate_size=16, vocab_size=16,
        rms_norm_eps=1e-6, hidden_act="silu",
    )
    monkeypatch.setattr(cp.current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setattr(cp.current_omni_platform, "supports_torch_inductor", lambda: True)
    monkeypatch.setattr(cp.envs, "VLLM_BATCH_INVARIANT", invariant)
    model = cp.CodePredictorWrapper(
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2)),
        cp_config=config,
        wrapper_config=cp.CodePredictorWrapperConfig(
            use_cuda_graphs=graphs, use_parallel_embedding=False,
            use_projection=False, return_proj_buf=True, sampling_mode="stored"),
    )
    calls = []

    def compile_model(fn, **kwargs):
        assert not invariant, "batch-invariant operators must not be lowered by standalone Inductor"
        calls.append("compile")
        assert kwargs == {"dynamic": False, "options": {"epilogue_fusion": False}}
        return fn

    monkeypatch.setattr(torch, "compile", compile_model)
    monkeypatch.setattr(model, "_warmup_buckets", lambda: calls.append("warmup"))
    monkeypatch.setattr(model, "_capture_cuda_graphs", lambda: calls.append("capture"))
    model._setup_compile()
    first = model._compiled_model_fwd
    model._setup_compile()
    assert model._compiled_model_fwd is first
    assert calls == ([] if invariant else ["compile"]) + ["warmup"] + (["capture"] if graphs else [])
    assert model._model_dtype == next(model.model.parameters()).dtype
    assert model._codec_embeds_list == list(model.model.codec_embedding)
