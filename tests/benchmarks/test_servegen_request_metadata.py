from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks.data_modules.random_multi_modal_dataset import (
    ServeGenSampleRequest,
)
from vllm_omni.benchmarks.patch.patch import (
    _attach_servegen_to_request_func_input,
)


def test_servegen_request_overrides_output_path_and_attaches_slo_metadata():
    sample = ServeGenSampleRequest(
        prompt="synthetic",
        prompt_len=16,
        expected_output_len=32,
        request_id="audio-7",
        output_modalities=["text", "audio"],
        slo_ms=2500.0,
        request_path="audio",
        predicted_stage_ms={"0": 100.0, "1": 75.0, "2": 350.0},
    )
    request = RequestFuncInput(
        prompt=sample.prompt,
        prompt_len=sample.prompt_len,
        output_len=sample.expected_output_len,
        request_id=sample.request_id,
        model="test-model",
        model_name="test-model",
        api_url="http://localhost/v1/chat/completions",
        extra_body={"modalities": ["text"], "stream": True},
    )

    _attach_servegen_to_request_func_input(sample, request)

    assert request.extra_body == {
        "modalities": ["text", "audio"],
        "stream": True,
    }
    assert request.servegen_slo_ms == 2500.0
    assert request.servegen_request_path == "audio"
    assert request.servegen_predicted_stage_ms == {
        "0": 100.0,
        "1": 75.0,
        "2": 350.0,
    }
