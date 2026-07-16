from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)


@pytest.mark.parametrize("model_mode", ["ar", "generation"])
def test_chunk_poll_keeps_policy_metadata_when_runtime_payload_replaces_it(
    model_mode,
):
    adapter = OmniChunkTransferAdapter.__new__(OmniChunkTransferAdapter)
    adapter.connector = SimpleNamespace(
        stage_id=2,
        get=lambda *args, **kwargs: (
            {
                "codes": {"audio": [1]},
                "meta": {
                    "finished": torch.tensor(True, dtype=torch.bool),
                    "sched_deadline_monotonic_s": -1.0,
                },
            },
            8,
        ),
    )
    adapter.model_mode = model_mode
    adapter.get_req_chunk = defaultdict(int)
    adapter.request_ids_mapping = {}
    adapter.request_payload = {}
    adapter.finished_requests = set()
    adapter._finished_load_reqs = set()

    request = SimpleNamespace(
        request_id="engine-1",
        status=RequestStatus.WAITING,
        prompt_token_ids=[],
        num_computed_tokens=0,
        additional_information={
            "meta": {
                "sched_source_request_id": "source-1",
                "sched_ingress_order": 7,
                "sched_deadline_monotonic_s": 123.0,
            }
        },
    )

    assert adapter._poll_single_request(request) is True
    metadata = request.additional_information["meta"]
    assert metadata["sched_source_request_id"] == "source-1"
    assert metadata["sched_ingress_order"] == 7
    assert metadata["sched_deadline_monotonic_s"] == 123.0
