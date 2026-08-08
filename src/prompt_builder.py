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

You must output an Evidence Memo in the following markdown format:

Confidence: [State the exact Confidence Level provided in the Evidence Score above] (e.g. High / Moderate / Low)

Evidence:
- [Fact / ticket count / sentiment details, or state "No validated customer evidence was found for this proposal."]

Missing:
- [Specify what data is missing: e.g., customer requests, usage analytics, sales feedback, engineering complexity]

Recommendation:
- [Clear action-oriented recommendation strictly following the Recommendation Guidelines above]
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
        formatted_list.append(f"Ticket ID: {ticket_id}\nTopic: {topic}\nMessage: {msg}\n---")

    return "\n".join(formatted_list)


def format_scoring_summary(scoring_info: dict = None) -> str:
    """
    Format scoring_info dictionary into a readable summary string.
    Handles the no-evidence scenario when a 'reason' key is present.
    """
    if not scoring_info or not isinstance(scoring_info, dict):
        return "Confidence Level: Not Provided | Evidence Score: N/A"

    confidence = scoring_info.get("confidence", "N/A")
    score = scoring_info.get("score", "N/A")
    reason = scoring_info.get("reason", None)
    factors = scoring_info.get("factors", {})

    summary_lines = [
        f"Confidence Level: {confidence}",
        f"Evidence Score: {score} / 100",
    ]

    if reason:
        summary_lines.append(f"Reason: {reason}")

    if factors:
        summary_lines.append("Factor Breakdown:")
        summary_lines.append(f"- Ticket Volume: {factors.get('ticket_volume', 0)} / 30 pts")
        summary_lines.append(f"- Severity & Urgency: {factors.get('severity', 0)} / 20 pts")
        summary_lines.append(f"- Sentiment Consistency: {factors.get('sentiment_consistency', 0)} / 25 pts")
        summary_lines.append(f"- Recency: {factors.get('recency', 0)} / 15 pts")
        summary_lines.append(f"- User Diversity: {factors.get('diversity', 0)} / 10 pts")

    return "\n".join(summary_lines)


def build_evidence_prompt(
    proposal: str,
    df_tickets: pd.DataFrame,
    scoring_info: dict = None,
    template_path: str = None,
) -> str:
    """
    Construct the analysis prompt for Gemini by loading the prompt template
    and substituting the feature proposal, scoring summary, and formatted support tickets.

    Parameters:
        proposal (str): The proposed feature description.
        df_tickets (pd.DataFrame): The DataFrame of support tickets.
        scoring_info (dict): Optional scoring information dictionary from EvidenceScoringEngine.
        template_path (str): Optional path to load the prompt template from a file.

    Returns:
        str: The fully formatted prompt.

    Raises:
        TypeError: If df_tickets is not a pandas DataFrame.
        ValueError: If proposal is empty, or df_tickets is missing required columns.
        FileNotFoundError: If template_path is provided but file does not exist.
        KeyError: If template is missing required placeholders.
    """
    # Validate inputs
    if not isinstance(proposal, str) or not proposal.strip():
        raise ValueError("proposal must be a non-empty string")

    if not isinstance(df_tickets, pd.DataFrame):
        raise TypeError("df_tickets must be a pandas DataFrame")

    if not df_tickets.empty:
        required_cols = {"topic", "message"}
        missing_cols = required_cols - set(df_tickets.columns)
        if missing_cols:
            raise ValueError(f"df_tickets is missing required columns: {missing_cols}")

    # Format the tickets and scoring summary to string representations
    tickets_text = format_tickets(df_tickets)
    scoring_summary = format_scoring_summary(scoring_info)

    # Load template
    template = DEFAULT_TEMPLATE
    if template_path:
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

    # Format template
    try:
        # Support both new template (with scoring_summary) and legacy templates (without scoring_summary)
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
