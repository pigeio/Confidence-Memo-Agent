import os
import pandas as pd

DEFAULT_TEMPLATE = """You are an honest, calibrated evidence analyst.
Your job is to evaluate customer support data to determine whether it supports a proposed feature or change.

Feature Proposal:
{proposal}

Support Tickets Evidence Base:
{tickets_text}

Analyze the tickets against the proposal. You must output an Evidence Memo in the following markdown format:

Confidence: [Low / Moderate / High] (Be highly calibrated: only select High if the evidence is clear, consistent, and represents a significant portion of tickets; Moderate if there is some signal but lacks breadth or is mixed; Low if there is very little or no signal).

Evidence:
- [Fact / ticket count / sentiment details]
- [Details from tickets]

Missing:
- [Specify what data is missing: e.g., analytics, usage data, engineering complexity, sales data]

Recommendation:
- [Clear action-oriented recommendation based on the evidence]
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


def build_evidence_prompt(proposal: str, df_tickets: pd.DataFrame, template_path: str = None) -> str:
    """
    Construct the analysis prompt for Gemini by loading the prompt template
    and substituting the feature proposal and formatted support tickets.

    Parameters:
        proposal (str): The proposed feature description.
        df_tickets (pd.DataFrame): The DataFrame of support tickets.
        template_path (str): Optional path to load the prompt template from a file.

    Returns:
        str: The fully formatted prompt.

    Raises:
        TypeError: If df_tickets is not a pandas DataFrame.
        ValueError: If proposal is empty, or df_tickets is missing required columns.
        FileNotFoundError: If template_path is provided but file does not exist.
        KeyError: If template is missing {proposal} or {tickets_text} placeholders.
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

    # Format the tickets to string representation
    tickets_text = format_tickets(df_tickets)

    # Load template
    template = DEFAULT_TEMPLATE
    if template_path:
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

    # Format template
    try:
        prompt = template.format(proposal=proposal, tickets_text=tickets_text)
    except KeyError as e:
        raise KeyError(f"Template is missing required placeholder: {e}")

    return prompt
