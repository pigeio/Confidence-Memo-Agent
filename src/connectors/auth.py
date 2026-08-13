import os
from src.connectors.exceptions import AuthenticationError


def resolve_credential(
    env_var: str,
    explicit_value: str = None,
    label: str = "credential",
) -> str:
    """
    Resolve a credential value using a clear priority chain:
    1. Explicit value passed directly (e.g. constructor argument).
    2. Environment variable (loaded from .env via python-dotenv).

    Parameters:
        env_var (str): Name of the environment variable to check.
        explicit_value (str | None): Value passed directly by the caller.
        label (str): Human-readable name for error messages (e.g. "Zendesk API token").

    Returns:
        str: The resolved credential value.

    Raises:
        AuthenticationError: If neither explicit value nor environment variable is set.
    """
    value = explicit_value or os.getenv(env_var)
    if not value:
        raise AuthenticationError(
            f"Missing {label}. Provide it directly or set the "
            f"{env_var} environment variable in your .env file."
        )
    return value
