"""Pure bounded-wait fairness state transitions."""

from __future__ import annotations

from dataclasses import dataclass


def _nonnegative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ProtectedAdmissionQuotaState:
    base_admissions_since_promotion: int

    def __post_init__(self) -> None:
        _nonnegative_int(
            self.base_admissions_since_promotion,
            field="base_admissions_since_promotion",
        )

    @classmethod
    def initial(cls, min_base_admissions: int) -> ProtectedAdmissionQuotaState:
        return cls(
            base_admissions_since_promotion=_nonnegative_int(
                min_base_admissions,
                field="min_base_admissions",
            )
        )


@dataclass(frozen=True, slots=True)
class ProtectedAdmissionQuotaDecision:
    promotion_allowed: bool
    quota_enforced: bool
    has_protected: bool
    has_unprotected: bool
    base_admissions_since_promotion: int
    min_base_admissions: int


def decide_protected_promotion(
    *,
    min_base_admissions: int,
    state: ProtectedAdmissionQuotaState,
    has_protected: bool,
    has_unprotected: bool,
) -> ProtectedAdmissionQuotaDecision:
    gap = _nonnegative_int(min_base_admissions, field="min_base_admissions")
    if not isinstance(has_protected, bool) or not isinstance(has_unprotected, bool):
        raise ValueError("backlog flags must be booleans")
    two_sided = has_protected and has_unprotected
    allowed = has_protected and (
        not has_unprotected
        or state.base_admissions_since_promotion >= gap
    )
    return ProtectedAdmissionQuotaDecision(
        promotion_allowed=allowed,
        quota_enforced=two_sided,
        has_protected=has_protected,
        has_unprotected=has_unprotected,
        base_admissions_since_promotion=state.base_admissions_since_promotion,
        min_base_admissions=gap,
    )


def advance_protected_quota(
    *,
    min_base_admissions: int,
    state: ProtectedAdmissionQuotaState,
    decision: ProtectedAdmissionQuotaDecision,
    admitted: bool,
    admitted_protected: bool,
) -> ProtectedAdmissionQuotaState:
    gap = _nonnegative_int(min_base_admissions, field="min_base_admissions")
    if not isinstance(admitted, bool) or not isinstance(admitted_protected, bool):
        raise ValueError("admission flags must be booleans")
    if admitted_protected and not admitted:
        raise ValueError("admitted_protected requires admitted")
    if not admitted:
        return state
    if (
        admitted_protected
        and decision.quota_enforced
        and decision.promotion_allowed
    ):
        return ProtectedAdmissionQuotaState(base_admissions_since_promotion=0)
    if admitted_protected and not decision.quota_enforced:
        # Work-conserving bypass does not consume credit. If an unprotected
        # request arrives next tick, it should not create an artificial gap.
        return state
    return ProtectedAdmissionQuotaState(
        base_admissions_since_promotion=min(
            gap,
            state.base_admissions_since_promotion + 1,
        )
    )
