import pytest
from src.decision_engine import (
    DecisionEngine,
    RecommendationAction,
    _normalize_level,
)


def test_normalize_level():
    assert _normalize_level("Low") == 1.5
    assert _normalize_level("Medium") == 3.0
    assert _normalize_level("High") == 4.5
    assert _normalize_level("Critical") == 5.0
    assert _normalize_level(4) == 4.0
    assert _normalize_level("unknown", default=2.5) == 2.5


def test_decision_engine_input_validation():
    engine = DecisionEngine()
    with pytest.raises(TypeError):
        engine.evaluate_decision(evidence_score="invalid_score")


def test_zero_evidence_short_circuit():
    engine = DecisionEngine()
    result = engine.evaluate_decision(
        evidence_score=0,
        engineering_effort="Medium",
        business_impact="High",
    )
    assert result.recommendation == RecommendationAction.INSUFFICIENT_EVIDENCE.value
    assert result.evidence_score == 0
    assert len(result.rationale) > 0
    assert "insufficient" in result.recommendation.lower() or "insufficient" in str(result.rationale).lower()


def test_high_impact_moderate_evidence_validate_further():
    engine = DecisionEngine()
    # High impact (4.5) with moderate evidence (55)
    result = engine.evaluate_decision(
        evidence_score=55,
        engineering_effort="Medium",
        business_impact="Critical",
        strategic_alignment="High",
    )
    assert result.recommendation == RecommendationAction.VALIDATE_FURTHER.value
    assert result.decision_tier in {"Strong Candidate", "Top Priority", "Conditional"}
    assert any("High business impact" in r for r in result.rationale)


def test_high_evidence_proceed_to_build():
    engine = DecisionEngine()
    # High evidence (85), low effort (1.5), high impact (4.5), high alignment (4.5)
    result = engine.evaluate_decision(
        evidence_score=85,
        engineering_effort="Low",
        business_impact="High",
        strategic_alignment="High",
        cost="Low",
        risk="Low",
    )
    assert result.recommendation == RecommendationAction.PROCEED_TO_BUILD.value
    assert result.priority_score >= 70
    assert result.decision_tier in {"Top Priority", "Strong Candidate"}
    assert len(result.assumptions) >= 2
    assert len(result.trade_offs) >= 1


def test_high_effort_risk_moderate_evidence_prototype_spike():
    engine = DecisionEngine()
    result = engine.evaluate_decision(
        evidence_score=60,
        engineering_effort="Very High",
        business_impact="Medium",
        strategic_alignment="Medium",
        risk="High",
        cost="High",
    )
    assert result.recommendation == RecommendationAction.PROTOTYPE_OR_SPIKE.value
    assert any("prototype" in r.lower() or "spike" in r.lower() for r in result.rationale)


def test_low_evidence_low_impact_deprioritize():
    engine = DecisionEngine()
    result = engine.evaluate_decision(
        evidence_score=25,
        engineering_effort="High",
        business_impact="Low",
        strategic_alignment="Low",
    )
    assert result.recommendation == RecommendationAction.DEPRIORITIZE.value


def test_excessive_risk_low_alignment_reject():
    engine = DecisionEngine()
    result = engine.evaluate_decision(
        evidence_score=20,
        engineering_effort="High",
        business_impact="Low",
        strategic_alignment="Low",
        risk="Critical",
    )
    assert result.recommendation == RecommendationAction.REJECT.value


def test_criteria_breakdown_and_to_dict():
    engine = DecisionEngine()
    result = engine.evaluate_decision(
        evidence_score=75,
        engineering_effort="Medium",
        business_impact="High",
        strategic_alignment="High",
    )
    res_dict = result.to_dict()
    assert "recommendation" in res_dict
    assert "priority_score" in res_dict
    assert "criteria_breakdown" in res_dict
    assert res_dict["criteria_breakdown"]["evidence_score"] == 75.0
