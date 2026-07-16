from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    _update_payload_common,
)

from vllm_omni.benchmarks.data_modules.random_multi_modal_dataset import (
    ServeGenSampleRequest,
)
from vllm_omni.benchmarks.patch.patch import (
    _attach_servegen_to_request_func_input,
)


def test_servegen_request_writes_output_path_and_scheduling_metadata_to_body():
    sample = ServeGenSampleRequest(
        prompt="synthetic",
        prompt_len=16,
        expected_output_len=32,
        request_id="audio-7",
        source_request_id="trace-audio-7",
        ingress_order=7,
        output_modalities=["text", "audio"],
        slo_ms=2500.0,
        request_path="audio",
        predicted_stage_ms=[100.0, 75.0, 350.0],
        predicted_stage_work_units=[64.0, 200.0, 1000.0],
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
        "omni_scheduling": {
            "schema_version": 1,
            "source_request_id": "trace-audio-7",
            "ingress_order": 7,
            "slo_ms": 2500.0,
            "request_path": "audio",
            "predicted_stage_ms": [100.0, 75.0, 350.0],
            "predicted_stage_work_units": [64.0, 200.0, 1000.0],
        },
    }

    payload: dict = {}
    _update_payload_common(payload, request)
    assert payload["omni_scheduling"] == request.extra_body["omni_scheduling"]
