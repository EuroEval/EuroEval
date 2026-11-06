"""Network access shared by dataset-creation scripts."""

import requests


def download_json_records(base_url: str, split: str) -> list[dict]:
    """Fetch one JSON split from a dataset source.

    Args:
        base_url:
            URL prefix ending at the dataset directory.
        split:
            Source split filename without the ``.json`` suffix.

    Returns:
        The decoded source records.
    """
    response = requests.get(f"{base_url}/{split}.json", timeout=30)
    response.raise_for_status()
    return response.json()
