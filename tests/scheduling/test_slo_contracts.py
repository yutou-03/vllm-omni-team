from types import MappingProxyType

import pytest

from vllm_omni.scheduling.contracts import (
    RequestPath,
    RequestSLOContract,
    SLOMilestone,
)


def test_request_slo_contract_is_deeply_immutable():
    deadlines = {
        SLOMilestone.TTFT: 101.0,
        SLOMilestone.E2E: 105.0,
    }
    contract = RequestSLOContract(
        schema_version=2,
        logical_request_id="chatcmpl-1",
        ingress_monotonic_s=100.0,
        request_path=RequestPath.TEXT,
        service_class="interactive",
        milestone_deadlines_s=deadlines,
        text_consume_rate_tps=20.0,
        audio_consume_rate=None,
        max_preemptions=1,
    )

    deadlines[SLOMilestone.E2E] = 999.0
    assert contract.milestone_deadlines_s[SLOMilestone.E2E] == 105.0
    assert isinstance(contract.milestone_deadlines_s, MappingProxyType)
    with pytest.raises(TypeError):
        contract.milestone_deadlines_s[SLOMilestone.E2E] = 106.0
    with pytest.raises(AttributeError):
        contract.max_preemptions = 2


@pytest.mark.parametrize(
    ("path", "deadlines", "message"),
    [
        ("text", {"e2e": 105.0}, "missing required milestones.*ttft"),
        (
            "audio",
            {"ttft": 101.0, "e2e": 105.0},
            "missing required milestones.*ttfp",
        ),
        (
            "text",
            {"ttft": 101.0, "ttfp": 102.0, "e2e": 105.0},
            "milestones not applicable to text.*ttfp",
        ),
    ],
)
def test_request_slo_contract_rejects_invalid_path_milestones(
    path, deadlines, message
):
    with pytest.raises(ValueError, match=message):
        RequestSLOContract(
            schema_version=2,
            logical_request_id="chatcmpl-1",
            ingress_monotonic_s=100.0,
            request_path=path,
            service_class="interactive",
            milestone_deadlines_s=deadlines,
            text_consume_rate_tps=None,
            audio_consume_rate=None,
            max_preemptions=1,
        )


def test_request_slo_contract_rejects_deadline_before_ingress():
    with pytest.raises(ValueError, match="ttft deadline must be after ingress"):
        RequestSLOContract(
            schema_version=2,
            logical_request_id="chatcmpl-1",
            ingress_monotonic_s=100.0,
            request_path="text",
            service_class="interactive",
            milestone_deadlines_s={"ttft": 99.0, "e2e": 105.0},
            text_consume_rate_tps=None,
            audio_consume_rate=None,
            max_preemptions=1,
        )


def test_request_slo_contract_round_trips_canonical_server_metadata():
    contract = RequestSLOContract.from_server_metadata(
        {
            "sched_schema_version": 2,
            "sched_logical_request_id": "chatcmpl-1",
            "sched_ingress_monotonic_s": 100.0,
            "sched_request_path": "audio",
            "sched_service_class": "realtime",
            "sched_milestone_deadlines_monotonic_s": {
                "ttft": 101.0,
                "ttfp": 102.0,
                "audio_gap": 100.2,
                "e2e": 105.0,
            },
            "sched_text_consume_rate_tps": 25.0,
            "sched_audio_consume_rate": 1.0,
            "sched_max_preemptions": 2,
        }
    )

    assert contract.request_path is RequestPath.AUDIO
    assert contract.milestone_deadlines_s[SLOMilestone.TTFP] == 102.0
    assert contract.to_server_metadata()["sched_milestone_deadlines_monotonic_s"] == {
        "ttft": 101.0,
        "ttfp": 102.0,
        "audio_gap": 100.2,
        "e2e": 105.0,
    }
