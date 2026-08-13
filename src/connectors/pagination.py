import requests
from src.connectors.exceptions import PaginationError, ConnectorError


def paginate_offset(
    url: str,
    headers: dict,
    params: dict = None,
    batch_size: int = 100,
    timeout: int = 30,
    results_key: str = "results",
) -> list[dict]:
    """
    Collect all records from an offset/page-based paginated API.

    Increments a `page` parameter starting at 1 until no more results are returned.
    Used by APIs like Zendesk that use ?page=1, ?page=2, etc.

    Parameters:
        url (str): Base API endpoint URL.
        headers (dict): HTTP headers (including auth).
        params (dict | None): Additional query parameters.
        batch_size (int): Number of records per page (sent as `per_page` parameter).
        timeout (int): HTTP request timeout in seconds.
        results_key (str): JSON key containing the list of records in each response.

    Returns:
        list[dict]: All collected records across all pages.

    Raises:
        PaginationError: If an HTTP error occurs during pagination.
    """
    all_results = []
    page = 1
    request_params = dict(params or {})
    request_params["per_page"] = batch_size

    while True:
        request_params["page"] = page
        try:
            response = requests.get(
                url, headers=headers, params=request_params, timeout=timeout
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise PaginationError(
                f"Offset pagination failed at page {page} for {url}: {e}"
            ) from e

        data = response.json()
        records = data.get(results_key, [])
        if not records:
            break

        all_results.extend(records)

        # Stop if we received fewer records than the batch size (last page)
        if len(records) < batch_size:
            break

        page += 1

    return all_results


def paginate_cursor(
    url: str,
    headers: dict,
    body: dict = None,
    batch_size: int = 100,
    timeout: int = 30,
    results_key: str = "results",
    cursor_path: tuple[str, ...] = ("next_cursor",),
    page_size_key: str = "page_size",
    method: str = "GET",
) -> list[dict]:
    """
    Collect all records from a cursor-based paginated API.

    Follows a cursor token across pages until the cursor is exhausted.
    Used by APIs like Intercom and Notion that return a `next_cursor` or
    `starting_after` token.

    Parameters:
        url (str): Base API endpoint URL.
        headers (dict): HTTP headers (including auth).
        body (dict | None): Request body (for POST-based pagination like Notion).
        batch_size (int): Number of records per page.
        timeout (int): HTTP request timeout in seconds.
        results_key (str): JSON key containing the list of records in each response.
        cursor_path (tuple[str, ...]): Dot-path to the next cursor value in the response
                                       JSON (e.g., ("pages", "next", "starting_after")).
        page_size_key (str): Key name for the page size parameter.
        method (str): HTTP method — "GET" or "POST".

    Returns:
        list[dict]: All collected records across all pages.

    Raises:
        PaginationError: If an HTTP error occurs during pagination.
    """
    all_results = []
    request_body = dict(body or {})
    request_body[page_size_key] = batch_size

    while True:
        try:
            if method.upper() == "POST":
                response = requests.post(
                    url, headers=headers, json=request_body, timeout=timeout
                )
            else:
                response = requests.get(
                    url, headers=headers, params=request_body, timeout=timeout
                )
            response.raise_for_status()
        except requests.RequestException as e:
            raise PaginationError(
                f"Cursor pagination failed for {url}: {e}"
            ) from e

        data = response.json()
        records = data.get(results_key, [])
        all_results.extend(records)

        # Navigate cursor_path to find the next cursor value
        cursor_value = data
        for key in cursor_path:
            if isinstance(cursor_value, dict):
                cursor_value = cursor_value.get(key)
            else:
                cursor_value = None
                break

        if not cursor_value:
            break

        # Inject cursor into next request
        request_body["start_cursor"] = cursor_value

    return all_results
