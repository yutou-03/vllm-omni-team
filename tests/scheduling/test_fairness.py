from __future__ import annotations

from vllm_omni.scheduling.fairness import (
    ProtectedAdmissionQuotaState,
    advance_protected_quota,
    decide_protected_promotion,
)


def test_two_sided_quota_opens_then_requires_exact_base_admission_gap():
    gap = 19
    state = ProtectedAdmissionQuotaState.initial(gap)
    decision = decide_protected_promotion(
        min_base_admissions=gap,
        state=state,
        has_protected=True,
        has_unprotected=True,
    )
    assert decision.promotion_allowed is True
    assert decision.quota_enforced is True

    state = advance_protected_quota(
        min_base_admissions=gap,
        state=state,
        decision=decision,
        admitted=True,
        admitted_protected=True,
    )
    assert state.base_admissions_since_promotion == 0
    for index in range(gap):
        decision = decide_protected_promotion(
            min_base_admissions=gap,
            state=state,
            has_protected=True,
            has_unprotected=True,
        )
        assert decision.promotion_allowed is (index == gap)
        state = advance_protected_quota(
            min_base_admissions=gap,
            state=state,
            decision=decision,
            admitted=True,
            admitted_protected=False,
        )
    decision = decide_protected_promotion(
        min_base_admissions=gap,
        state=state,
        has_protected=True,
        has_unprotected=True,
    )
    assert state.base_admissions_since_promotion == gap
    assert decision.promotion_allowed is True


def test_only_protected_waiters_bypass_quota_without_consuming_credit():
    state = ProtectedAdmissionQuotaState(base_admissions_since_promotion=3)
    decision = decide_protected_promotion(
        min_base_admissions=19,
        state=state,
        has_protected=True,
        has_unprotected=False,
    )

    assert decision.promotion_allowed is True
    assert decision.quota_enforced is False
    assert advance_protected_quota(
        min_base_admissions=19,
        state=state,
        decision=decision,
        admitted=True,
        admitted_protected=True,
    ) == state


def test_no_protected_waiter_does_not_open_promotion_and_saturates_credit():
    state = ProtectedAdmissionQuotaState(base_admissions_since_promotion=18)
    decision = decide_protected_promotion(
        min_base_admissions=19,
        state=state,
        has_protected=False,
        has_unprotected=True,
    )

    assert decision.promotion_allowed is False
    next_state = advance_protected_quota(
        min_base_admissions=19,
        state=state,
        decision=decision,
        admitted=True,
        admitted_protected=False,
    )
    assert next_state.base_admissions_since_promotion == 19
