import os
import pandas as pd

DEFAULT_TEMPLATE = """You are an honest, calibrated evidence analyst.
Your job is to evaluate customer support data and explain the pre-calculated Evidence Score for a proposed feature or change.

Feature Proposal:
{proposal}

Deterministically Calculated Evidence Score:
{scoring_summary}

Support Tickets Evidence Base:
{tickets_text}

IMPORTANT INSTRUCTIONS:
1. Do NOT calculate or invent a new Confidence Level or Evidence Score. You must accept and explain the provided Confidence Level and Evidence Score.

2. RECOMMENDATION GUIDELINES (STRICT COMPLIANCE REQUIRED):
- High Evidence (Score 70-100): Recommend proceeding with prioritization if aligned with business goals.
- Moderate Evidence (Score 40-69): Recommend investigating further before prioritization.
- Low Evidence with Relevant Tickets (Score 1-39): Recommend collecting more customer evidence before prioritization.
- No Validated Evidence (Score 0): Do NOT make a build or reject/deprioritize recommendation! Explicitly state that there is currently insufficient validated evidence to support or reject this proposal, and recommend collecting additional customer feedback, analytics, or research before prioritization.

3. CRITICAL RULE:
Absence of evidence is NOT evidence of absence. Never interpret missing customer support tickets as evidence against a feature, and never recommend rejecting or deprioritizing a feature solely because no support tickets were found.

You must output an Evidence Memo in the following exact markdown structure (do NOT change section order or headings):

Confidence: [State exact Confidence Level] (Score: [State exact Evidence Score] / 100)

Evidence:
- [Fact / ticket count / sentiment details, or state "No validated customer evidence was found for this proposal."]

Missing:
- [Specify what data is missing: e.g., customer requests, usage analytics, sales feedback, engineering complexity]

Recommendation:
- [Clear action-oriented recommendation strictly following the Recommendation Guidelines above]
"""

DEFAULT_DECISION_TEMPLATE = """You are an executive product evidence analyst.
Your job is to synthesize customer feedback, strategic criteria, and the pre-calculated deterministic Decision Engine results into a clear, calibrated Decision Memo.

Feature Proposal:
{proposal}

Deterministically Computed Decision & Priority:
{decision_summary}

Evidence Base & Telemetry (Deduplicated & Clustered):
{evidence_summary}

Representative Customer Tickets:
{tickets_text}

Deterministic Strategic Analysis:
Assumptions:
{assumptions_text}

Risks:
{risks_text}

Trade-Offs:
{trade_offs_text}

Missing Information:
{missing_info_text}

Deterministic Recommendation Rationale:
{rationale_text}

CRITICAL RULES:
1. Do NOT invent, override, or modify the pre-calculated Recommendation, Priority Score, or Evidence Score. Accept and clearly explain the deterministic results.
2. Absence of evidence is NOT evidence of absence: never recommend rejecting a proposal solely because customer support tickets are missing.
3. You must output the Decision Memo in the following exact markdown structure:

# Decision Memo: [Feature Proposal Title]

**Executive Recommendation:** [State exact Recommendation Action]
**Priority Score:** [Priority Score] / 100 ([Decision Tier])
**Evidence Score:** [Evidence Score] / 100 ([Confidence Level])

---

### 1. Recommendation Rationale
- [Explain the key reasons supporting the deterministic recommendation based on the rationale provided]

### 2. Customer Evidence Analysis
- [Synthesize customer sentiment, ticket clusters, volume, and deduplication insights]
- [Reference specific customer quotes or representative ticket IDs]

### 3. Key Assumptions
- [List verified assumptions and baseline conditions]

### 4. Risks & Mitigations
- [List technical, market, operational, or evidence gap risks]

### 5. Strategic Trade-Offs
- [Summarize trade-offs between effort, impact, urgency, and opportunity costs]

### 6. Missing Information & Next Steps
- [Enumerate specific telemetry, research, or discovery needed]
"""


def format_tickets(df: pd.DataFrame) -> str:
    """
    Format a DataFrame of support tickets into a readable text format.
    Each ticket lists its ID, topic, and message.
    """
    if df.empty:
        return "No tickets found matching the keyword criteria."

    formatted_list = []
    for idx, row in df.iterrows():
        ticket_id = row.get("ticket_id", idx)
        topic = row.get("topic", "N/A")
        msg = row.get("message", "N/A")
        cluster_info = (
            f" [Cluster {row['cluster_id']}]" if "cluster_id" in row else ""
        )
        formatted_list.append(
            f"Ticket ID: {ticket_id}{cluster_info}\nTopic: {topic}\nMessage: {msg}\n---"
        )

    return "\n".join(formatted_list)


def format_scoring_summary(scoring_info: dict = None) -> str:
    """
    Format scoring_info dictionary into a readable summary string.
    Handles evidence_summary telemetry, score caps, and factor breakdowns.
    """
    if not scoring_info or not isinstance(scoring_info, dict):
        return "Confidence Level: Not Provided | Evidence Score: N/A"

    confidence = scoring_info.get("confidence", "N/A")
    score = scoring_info.get("score", "N/A")
    reason = scoring_info.get("reason", None)
    evidence_summary = scoring_info.get("evidence_summary", {})
    factors = scoring_info.get("factors", {})

    summary_lines = [
        f"Confidence Level: {confidence}",
        f"Evidence Score: {score} / 100",
    ]

    if reason:
        summary_lines.append(f"Reason: {reason}")

    if evidence_summary:
        summary_lines.append("Evidence Telemetry:")
        summary_lines.append(
            f"- Validated Tickets: {evidence_summary.get('validated_tickets', 0)}"
        )
        summary_lines.append(
            f"- Retrieved Candidates: {evidence_summary.get('retrieved_tickets', 0)}"
        )
        summary_lines.append(
            f"- Rejected Candidates: {evidence_summary.get('rejected_tickets', 0)}"
        )
        summary_lines.append(
            f"- Average Similarity: {evidence_summary.get('average_similarity', 0.0):.2f}"
        )
        summary_lines.append(
            f"- Sample Size Tier: {evidence_summary.get('sample_size', 'N/A')}"
        )

    if factors:
        summary_lines.append("Factor Breakdown:")
        summary_lines.append(
            f"- Ticket Volume: {factors.get('ticket_volume', 0)} / 30 pts"
        )
        summary_lines.append(
            f"- Severity & Urgency: {factors.get('severity', 0)} / 20 pts"
        )
        summary_lines.append(
            f"- Sentiment Consistency: {factors.get('sentiment_consistency', 0)} / 25 pts"
        )
        summary_lines.append(f"- Recency: {factors.get('recency', 0)} / 15 pts")
        summary_lines.append(f"- User Diversity: {factors.get('diversity', 0)} / 10 pts")

    return "\n".join(summary_lines)


def format_bullet_list(items: list[str] | None, default_empty: str = "None identified.") -> str:
    """Format list of strings into markdown bullet points."""
    if not items:
        return f"- {default_empty}"
    return "\n".join(f"- {item}" for item in items)


def format_decision_summary(decision_info: dict = None) -> str:
    """Format DecisionResult or decision dictionary into a readable summary string."""
    if not decision_info or not isinstance(decision_info, dict):
        return "Recommendation: N/A | Priority Score: N/A"

    rec = decision_info.get("recommendation", "N/A")
    pri_score = decision_info.get("priority_score", "N/A")
    tier = decision_info.get("decision_tier", "N/A")
    ev_score = decision_info.get("evidence_score", "N/A")
    breakdown = decision_info.get("criteria_breakdown", {})

    lines = [
        f"Recommendation: {rec}",
        f"Priority Score: {pri_score} / 100 ({tier})",
        f"Evidence Score: {ev_score} / 100",
    ]

    if breakdown:
        lines.append("Multi-Criteria Breakdown:")
        lines.append(f"- Engineering Effort: {breakdown.get('engineering_effort_level', 'N/A')}/5.0")
        lines.append(f"- Business Impact: {breakdown.get('business_impact_level', 'N/A')}/5.0")
        lines.append(f"- Strategic Alignment: {breakdown.get('strategic_alignment_level', 'N/A')}/5.0")
        lines.append(f"- Cost: {breakdown.get('cost_level', 'N/A')}/5.0")
        lines.append(f"- Risk: {breakdown.get('risk_level', 'N/A')}/5.0")

    return "\n".join(lines)


def format_evidence_intelligence_summary(
    scoring_info: dict = None,
    deduplication_stats: dict = None,
    clustering_stats: dict = None,
) -> str:
    """Format combined evidence intelligence telemetry including scoring, deduplication, and clustering."""
    lines = []
    base_summary = format_scoring_summary(scoring_info)
    lines.append(base_summary)

    if deduplication_stats:
        lines.append("\nDeduplication Telemetry:")
        lines.append(f"- Total Raw Tickets: {deduplication_stats.get('total_input_count', 0)}")
        lines.append(f"- Unique Retained Tickets: {deduplication_stats.get('unique_count', 0)}")
        lines.append(f"- Duplicates Removed: {deduplication_stats.get('duplicate_count', 0)} ({deduplication_stats.get('duplicate_rate', 0.0)*100:.1f}%)")
        lines.append(f"  * Exact Duplicates: {deduplication_stats.get('exact_duplicates_count', 0)}")
        lines.append(f"  * Semantic Duplicates: {deduplication_stats.get('semantic_duplicates_count', 0)}")

    if clustering_stats:
        lines.append("\nClustering & Theme Analysis:")
        lines.append(f"- Total Theme Clusters: {clustering_stats.get('total_clusters', 0)}")
        lines.append(f"- Average Cluster Size: {clustering_stats.get('average_cluster_size', 0.0):.1f} tickets")
        clusters = clustering_stats.get("clusters", [])
        for c in clusters[:5]:  # Top 5 clusters
            lines.append(f"  * [Cluster {c['cluster_id']}] (Size: {c['size']}, Cohesion: {c.get('intra_cluster_similarity', 1.0):.2f}) - {c.get('theme_label', 'Theme')}")

    return "\n".join(lines)


def build_evidence_prompt(
    proposal: str,
    df_tickets: pd.DataFrame,
    scoring_info: dict = None,
    template_path: str = None,
) -> str:
    """
    Construct the analysis prompt for Gemini for an Evidence Memo.
    """
    if not isinstance(proposal, str) or not proposal.strip():
        raise ValueError("proposal must be a non-empty string")

    if not isinstance(df_tickets, pd.DataFrame):
        raise TypeError("df_tickets must be a pandas DataFrame")

    if not df_tickets.empty:
        required_cols = {"topic", "message"}
        missing_cols = required_cols - set(df_tickets.columns)
        if missing_cols:
            raise ValueError(f"df_tickets is missing required columns: {missing_cols}")

    tickets_text = format_tickets(df_tickets)
    scoring_summary = format_scoring_summary(scoring_info)

    template = DEFAULT_TEMPLATE
    if template_path:
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

    try:
        if "{scoring_summary}" in template:
            prompt = template.format(
                proposal=proposal,
                scoring_summary=scoring_summary,
                tickets_text=tickets_text,
            )
        else:
            prompt = template.format(proposal=proposal, tickets_text=tickets_text)
    except KeyError as e:
        raise KeyError(f"Template is missing required placeholder: {e}")

    return prompt


def build_decision_prompt(
    proposal: str,
    df_tickets: pd.DataFrame,
    decision_info: dict,
    scoring_info: dict = None,
    deduplication_stats: dict = None,
    clustering_stats: dict = None,
    template_path: str = None,
) -> str:
    """
    Construct the executive prompt for Gemini for a Decision Memo.
    """
    if not isinstance(proposal, str) or not proposal.strip():
        raise ValueError("proposal must be a non-empty string")

    if not isinstance(df_tickets, pd.DataFrame):
        raise TypeError("df_tickets must be a pandas DataFrame")

    if not isinstance(decision_info, dict):
        raise TypeError("decision_info must be a dictionary or DecisionResult.to_dict()")

    if not df_tickets.empty:
        required_cols = {"topic", "message"}
        missing_cols = required_cols - set(df_tickets.columns)
        if missing_cols:
            raise ValueError(f"df_tickets is missing required columns: {missing_cols}")

    tickets_text = format_tickets(df_tickets)
    decision_summary = format_decision_summary(decision_info)
    evidence_summary = format_evidence_intelligence_summary(
        scoring_info, deduplication_stats, clustering_stats
    )

    assumptions_text = format_bullet_list(decision_info.get("assumptions"))
    risks_text = format_bullet_list(decision_info.get("risks"))
    trade_offs_text = format_bullet_list(decision_info.get("trade_offs"))
    missing_info_text = format_bullet_list(decision_info.get("missing_information"))
    rationale_text = format_bullet_list(decision_info.get("rationale"))

    template = DEFAULT_DECISION_TEMPLATE
    if template_path:
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

    try:
        prompt = template.format(
            proposal=proposal,
            decision_summary=decision_summary,
            evidence_summary=evidence_summary,
            tickets_text=tickets_text,
            assumptions_text=assumptions_text,
            risks_text=risks_text,
            trade_offs_text=trade_offs_text,
            missing_info_text=missing_info_text,
            rationale_text=rationale_text,
        )
    except KeyError as e:
        raise KeyError(f"Template is missing required placeholder: {e}")

    return prompt
