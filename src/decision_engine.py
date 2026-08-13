import logging
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class RecommendationAction(str, Enum):
    PROCEED_TO_BUILD = "PROCEED_TO_BUILD"
    VALIDATE_FURTHER = "VALIDATE_FURTHER"
    PROTOTYPE_OR_SPIKE = "PROTOTYPE_OR_SPIKE"
    DEPRIORITIZE = "DEPRIORITIZE"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# Scale mappings: converts text labels or numeric inputs to 1-5 float scale
_LEVEL_MAP = {
    "very low": 1.0,
    "low": 1.5,
    "medium": 3.0,
    "moderate": 3.0,
    "high": 4.5,
    "very high": 5.0,
    "critical": 5.0,
}


def _normalize_level(val: str | int | float, default: float = 3.0) -> float:
    """Normalize level string or number to a 1.0 - 5.0 float scale."""
    if isinstance(val, (int, float)):
        return float(max(1.0, min(5.0, val)))
    if isinstance(val, str):
        normalized = val.strip().lower()
        if normalized in _LEVEL_MAP:
            return _LEVEL_MAP[normalized]
        try:
            num = float(normalized)
            return float(max(1.0, min(5.0, num)))
        except ValueError:
            pass
    return default


@dataclass
class DecisionResult:
    """Structured deterministic decision output."""

    recommendation: str  # One of RecommendationAction
    priority_score: int  # 0 to 100
    decision_tier: str  # 'Top Priority', 'Strong Candidate', 'Conditional', 'Low Priority', 'Unfavorable'
    evidence_score: int  # 0 to 100
    rationale: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    trade_offs: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    criteria_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionEngine:
    """
    Deterministic Multi-Criteria Product Decision Engine.
    Combines evidence strength with engineering effort, business impact, strategic alignment,
    cost, and risk to compute an objective recommendation and priority score.
    """

    # Weights for Priority Score computation
    WEIGHT_EVIDENCE = 0.35
    WEIGHT_IMPACT = 0.30
    WEIGHT_ALIGNMENT = 0.15
    WEIGHT_EFFORT = 0.10
    WEIGHT_COST = 0.05
    WEIGHT_RISK = 0.05

    def evaluate_decision(
        self,
        evidence_score: int | float,
        engineering_effort: str | int | float = "Medium",
        business_impact: str | int | float = "Medium",
        strategic_alignment: str | int | float = "Medium",
        cost: str | int | float = "Low",
        risk: str | int | float = "Low",
        evidence_summary: dict | None = None,
        deduplication_stats: dict | None = None,
        clustering_stats: dict | None = None,
    ) -> DecisionResult:
        """
        Deterministically evaluate a product proposal across all business, engineering, and evidence criteria.

        Parameters:
            evidence_score (int | float): 0-100 deterministic evidence score from EvidenceScoringEngine.
            engineering_effort (str | int | float): Engineering complexity ('Low', 'Medium', 'High', 'Very High' or 1-5).
            business_impact (str | int | float): Potential business impact ('Low', 'Medium', 'High', 'Critical' or 1-5).
            strategic_alignment (str | int | float): Strategic fit with product goals ('Low', 'Medium', 'High' or 1-5).
            cost (str | int | float): Financial/infrastructure cost ('Low', 'Medium', 'High' or 1-5).
            risk (str | int | float): Technical or execution risk ('Low', 'Medium', 'High' or 1-5).
            evidence_summary (dict): Optional telemetry dictionary from EvidenceScoringEngine.
            deduplication_stats (dict): Optional deduplication stats from EvidenceDeduplicator.
            clustering_stats (dict): Optional cluster stats from EvidenceClusterer.

        Returns:
            DecisionResult: Structured deterministic decision and recommendation.
        """
        # Validate and clamp evidence score
        try:
            ev_score = float(evidence_score)
        except (TypeError, ValueError):
            raise TypeError("evidence_score must be a numeric value")
        ev_score = max(0.0, min(100.0, ev_score))

        # Normalize 1-5 criteria
        eff_lvl = _normalize_level(engineering_effort, default=3.0)
        imp_lvl = _normalize_level(business_impact, default=3.0)
        align_lvl = _normalize_level(strategic_alignment, default=3.0)
        cost_lvl = _normalize_level(cost, default=1.5)
        risk_lvl = _normalize_level(risk, default=1.5)

        # Scale 1-5 criteria to 0-100
        eff_norm = (eff_lvl - 1.0) / 4.0 * 100.0
        imp_norm = (imp_lvl - 1.0) / 4.0 * 100.0
        align_norm = (align_lvl - 1.0) / 4.0 * 100.0
        cost_norm = (cost_lvl - 1.0) / 4.0 * 100.0
        risk_norm = (risk_lvl - 1.0) / 4.0 * 100.0

        # Deterministic Priority Score:
        # Benefit = Evidence(0.35) + Impact(0.30) + Alignment(0.15)
        # Cost/Friction = Effort(0.10) + Cost(0.05) + Risk(0.05)
        raw_benefits = (
            self.WEIGHT_EVIDENCE * ev_score
            + self.WEIGHT_IMPACT * imp_norm
            + self.WEIGHT_ALIGNMENT * align_norm
        )
        raw_frictions = (
            self.WEIGHT_EFFORT * eff_norm
            + self.WEIGHT_COST * cost_norm
            + self.WEIGHT_RISK * risk_norm
        )

        priority_score = int(round(max(0.0, min(100.0, raw_benefits - raw_frictions + 20.0))))

        # Determine Recommendation Action via deterministic gating logic
        recommendation, rationale = self._determine_recommendation_and_rationale(
            ev_score=ev_score,
            eff_lvl=eff_lvl,
            imp_lvl=imp_lvl,
            align_lvl=align_lvl,
            cost_lvl=cost_lvl,
            risk_lvl=risk_lvl,
            priority_score=priority_score,
            evidence_summary=evidence_summary,
            deduplication_stats=deduplication_stats,
            clustering_stats=clustering_stats,
        )

        # Map priority score to decision tier
        if priority_score >= 80:
            decision_tier = "Top Priority"
        elif priority_score >= 60:
            decision_tier = "Strong Candidate"
        elif priority_score >= 40:
            decision_tier = "Conditional"
        elif priority_score >= 20:
            decision_tier = "Low Priority"
        else:
            decision_tier = "Unfavorable"

        # Deterministically build Assumptions, Risks, Trade-offs, and Missing Information
        assumptions = self._build_assumptions(
            eff_lvl, imp_lvl, align_lvl, deduplication_stats
        )
        risks = self._build_risks(risk_lvl, eff_lvl, cost_lvl, ev_score)
        trade_offs = self._build_trade_offs(
            imp_lvl, eff_lvl, align_lvl, cost_lvl, ev_score
        )
        missing_info = self._build_missing_info(
            ev_score, evidence_summary, deduplication_stats
        )

        criteria_breakdown = {
            "evidence_score": round(ev_score, 1),
            "engineering_effort_level": eff_lvl,
            "business_impact_level": imp_lvl,
            "strategic_alignment_level": align_lvl,
            "cost_level": cost_lvl,
            "risk_level": risk_lvl,
            "raw_benefits": round(raw_benefits, 2),
            "raw_frictions": round(raw_frictions, 2),
        }

        return DecisionResult(
            recommendation=recommendation.value,
            priority_score=priority_score,
            decision_tier=decision_tier,
            evidence_score=int(round(ev_score)),
            rationale=rationale,
            assumptions=assumptions,
            risks=risks,
            trade_offs=trade_offs,
            missing_information=missing_info,
            criteria_breakdown=criteria_breakdown,
        )

    def _determine_recommendation_and_rationale(
        self,
        ev_score: float,
        eff_lvl: float,
        imp_lvl: float,
        align_lvl: float,
        cost_lvl: float,
        risk_lvl: float,
        priority_score: int,
        evidence_summary: dict | None,
        deduplication_stats: dict | None,
        clustering_stats: dict | None,
    ) -> tuple[RecommendationAction, list[str]]:
        rationale = []

        # Zero evidence short-circuit
        if ev_score == 0:
            rationale.append(
                "No validated customer support evidence exists for this proposal."
            )
            rationale.append(
                "Absence of evidence is not evidence of absence; collect user feedback or telemetry before decision."
            )
            return RecommendationAction.INSUFFICIENT_EVIDENCE, rationale

        # Severe risk/cost gating
        if risk_lvl >= 4.5 and ev_score < 40 and align_lvl <= 2.0:
            rationale.append(
                "High execution risk paired with weak evidence and low strategic alignment."
            )
            rationale.append(
                "The ROI profile does not justify the disproportionate risk."
            )
            return RecommendationAction.REJECT, rationale

        # High Impact with Low/Moderate Evidence -> VALIDATE_FURTHER
        if imp_lvl >= 4.0 and ev_score < 70:
            rationale.append(
                f"High business impact potential ({imp_lvl:.1f}/5.0), but customer evidence strength is currently {ev_score:.0f}/100."
            )
            rationale.append(
                "High impact justifies active validation (surveys, interviews, analytics) before engineering commitment."
            )
            return RecommendationAction.VALIDATE_FURTHER, rationale

        # High Effort + High Risk with Moderate Evidence -> PROTOTYPE_OR_SPIKE
        if eff_lvl >= 4.0 and (risk_lvl >= 3.5 or cost_lvl >= 3.5) and (40 <= ev_score <= 75):
            rationale.append(
                "High technical complexity and execution risk require architectural de-risking."
            )
            rationale.append(
                f"Moderate customer evidence ({ev_score:.0f}/100) supports building a targeted prototype or technical spike."
            )
            return RecommendationAction.PROTOTYPE_OR_SPIKE, rationale

        # High Evidence + Positive Priority Score + Acceptable Risk -> PROCEED_TO_BUILD
        if ev_score >= 70 and priority_score >= 50 and risk_lvl <= 4.0:
            rationale.append(
                f"Strong, validated customer evidence ({ev_score:.0f}/100) with solid strategic and business alignment."
            )
            rationale.append(
                f"Priority score ({priority_score}/100) confirms a favorable return on engineering investment."
            )
            return RecommendationAction.PROCEED_TO_BUILD, rationale

        # Low Evidence + Low/Medium Impact -> DEPRIORITIZE
        if ev_score < 40 and imp_lvl <= 3.0:
            rationale.append(
                f"Low customer evidence ({ev_score:.0f}/100) combined with modest business impact ({imp_lvl:.1f}/5.0)."
            )
            rationale.append(
                "Allocate engineering capacity to higher-leverage roadmap initiatives."
            )
            return RecommendationAction.DEPRIORITIZE, rationale

        # High Effort + Low Alignment -> DEPRIORITIZE
        if eff_lvl >= 4.0 and align_lvl <= 2.5:
            rationale.append(
                "High engineering effort does not align closely with core strategic priorities."
            )
            return RecommendationAction.DEPRIORITIZE, rationale

        # Default fallback based on priority score
        if priority_score >= 60:
            rationale.append(
                f"Favorable multi-criteria balance with priority score of {priority_score}/100."
            )
            return RecommendationAction.PROCEED_TO_BUILD, rationale
        elif priority_score >= 40:
            rationale.append(
                "Borderline priority score indicates mixed signals; recommend targeted discovery."
            )
            return RecommendationAction.VALIDATE_FURTHER, rationale
        else:
            rationale.append(
                f"Sub-optimal priority score ({priority_score}/100); recommend deprioritizing."
            )
            return RecommendationAction.DEPRIORITIZE, rationale

    def _build_assumptions(
        self,
        eff_lvl: float,
        imp_lvl: float,
        align_lvl: float,
        dedup_stats: dict | None,
    ) -> list[str]:
        assumptions = [
            f"Assumes estimated engineering effort ({eff_lvl:.1f}/5.0) remains within scoped capacity.",
            f"Assumes business impact potential ({imp_lvl:.1f}/5.0) reflects addressable user demand.",
        ]
        if dedup_stats and dedup_stats.get("duplicate_count", 0) > 0:
            assumptions.append(
                f"Customer feedback volume was normalized for {dedup_stats['duplicate_count']} duplicate inquiries."
            )
        return assumptions

    def _build_risks(
        self,
        risk_lvl: float,
        eff_lvl: float,
        cost_lvl: float,
        ev_score: float,
    ) -> list[str]:
        risks = []
        if risk_lvl >= 3.5:
            risks.append(
                f"Execution Risk: High architectural or domain complexity ({risk_lvl:.1f}/5.0)."
            )
        if eff_lvl >= 4.0:
            risks.append(
                "Opportunity Cost: Significant engineering resource allocation may delay parallel roadmap tracks."
            )
        if cost_lvl >= 3.5:
            risks.append(
                f"Financial/Infrastructure Risk: Heightened recurring operational or third-party costs ({cost_lvl:.1f}/5.0)."
            )
        if ev_score < 50:
            risks.append(
                "Market Validation Risk: Limited customer support signals may lead to building a low-adoption feature."
            )
        if not risks:
            risks.append(
                "Manageable Risk Profile: No severe technical, operational, or evidence red flags identified."
            )
        return risks

    def _build_trade_offs(
        self,
        imp_lvl: float,
        eff_lvl: float,
        align_lvl: float,
        cost_lvl: float,
        ev_score: float,
    ) -> list[str]:
        trade_offs = []
        if imp_lvl >= 4.0 and eff_lvl >= 4.0:
            trade_offs.append(
                "High Impact vs High Effort: Requires large sprint investment for high anticipated user payoff."
            )
        elif imp_lvl >= 4.0 and eff_lvl <= 2.0:
            trade_offs.append(
                "High Leverage Quick-Win: High anticipated return with minimal engineering overhead."
            )
        elif imp_lvl <= 2.0 and eff_lvl >= 3.5:
            trade_offs.append(
                "Unfavorable Cost-to-Value: Significant engineering expenditure for marginal business gain."
            )

        if ev_score >= 70 and align_lvl <= 2.5:
            trade_offs.append(
                "Customer Demand vs Strategic Focus: Strong user demand exists but diverges from immediate strategic goals."
            )

        if not trade_offs:
            trade_offs.append(
                "Balanced Investment: Resource requirements align proportionally with expected utility."
            )
        return trade_offs

    def _build_missing_info(
        self,
        ev_score: float,
        evidence_summary: dict | None,
        dedup_stats: dict | None,
    ) -> list[str]:
        missing = []
        if ev_score < 70:
            missing.append(
                "Quantitative user analytics and product telemetry (e.g. drop-off rates, click tracking)."
            )
            missing.append(
                "Qualitative feedback from user interviews or customer advisory boards."
            )
        if evidence_summary and evidence_summary.get("sample_size") in {"Zero", "Small"}:
            missing.append(
                "Broader sample size of customer support tickets across diverse customer segments."
            )
        missing.append("Detailed technical specification and architecture RFC from engineering.")
        return missing
