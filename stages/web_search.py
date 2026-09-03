"""
stages/web_search.py
--------------------
Stage 2 - Reverse Image Search

Uses the SerpAPI Google Lens endpoint to perform a genuine reverse-image search
and returns structured social-media / web results.

API docs: https://serpapi.com/google-lens-api

Environment variables required (in .env):
    SERPAPI_KEY - your SerpAPI API key

Mock mode (--mock flag or MOCK_MODE=1):
    Returns a pre-defined result without hitting any external API.
"""

import json
import os
from pathlib import Path

import requests

from utils.logger import get_logger

log = get_logger("web_search")

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# Mock data for offline / demo use
MOCK_RESULT = {
    "engine": "google_lens [MOCK]",
    "search_url": "https://lens.google.com/search?p=MOCK",
    "results": [
        {
            "position": 1,
            "title": "Sample Person - LinkedIn Profile [MOCK]",
            "link": "https://www.linkedin.com/in/sample-person-mock",
            "source": "LinkedIn",
            "thumbnail": "https://via.placeholder.com/150",
        },
        {
            "position": 2,
            "title": "Sample Person on Twitter [MOCK]",
            "link": "https://twitter.com/sampleperson/status/123456789",
            "source": "Twitter / X",
            "thumbnail": "https://via.placeholder.com/150",
        },
    ],
    "best_match": {
        "position": 1,
        "title": "Sample Person - LinkedIn Profile [MOCK]",
        "link": "https://www.linkedin.com/in/sample-person-mock",
        "source": "LinkedIn",
        "thumbnail": "https://via.placeholder.com/150",
    },
}


class WebSearchError(Exception):
    """Raised when the reverse-image search returns no usable results."""


def _upload_image_for_url(image_path) -> str:
    """
    Upload a local image to a free temporary host and return its public URL.

    Tries 0x0.st first (no auth), then falls back to file.io.
    The URL is used to pass the image to SerpAPI Google Lens.
    """
    path = Path(image_path)

    # --- Primary host: 0x0.st (no account required) ---
    try:
        log.info("Hosting image on 0x0.st for URL-based search ...")
        with open(path, "rb") as f:
            resp = requests.post(
                "https://0x0.st",
                files={"file": (path.name, f, "image/jpeg")},
                timeout=30,
            )
        if resp.status_code == 200 and resp.text.strip().startswith("http"):
            url = resp.text.strip()
            log.info("Image hosted at: %s", url)
            return url
        log.warning("0x0.st returned unexpected response (%d), trying fallback ...", resp.status_code)
    except Exception as exc:
        log.warning("0x0.st upload failed (%s), trying fallback ...", exc)

    # --- Fallback host: file.io (free, expires after first download) ---
    try:
        log.info("Hosting image on file.io ...")
        with open(path, "rb") as f:
            resp = requests.post(
                "https://file.io",
                files={"file": (path.name, f, "image/jpeg")},
                timeout=30,
            )
        if resp.status_code == 200:
            data = resp.json()
            url = data.get("link", "")
            if url:
                log.info("Image hosted at: %s", url)
                return url
    except Exception as exc:
        log.warning("file.io upload also failed: %s", exc)

    raise WebSearchError(
        "Could not upload image to a temporary host for URL-based search. "
        "Check your internet connection."
    )


def reverse_image_search(image_path: str, mock: bool = False) -> dict:
    """
    Perform a reverse-image search for image_path using SerpAPI Google Lens.

    Parameters
    ----------
    image_path : str
        Path to the face image (or any image to reverse-search).
    mock : bool
        If True, skip the real API call and return MOCK_RESULT.

    Returns
    -------
    dict with keys:
        engine      - search engine description
        search_url  - URL that was searched
        results     - list of result dicts (position, title, link, source, thumbnail)
        best_match  - the top result dict
    """
    if mock or os.getenv("MOCK_MODE", "0") == "1":
        log.info("[MOCK] Returning pre-defined reverse-image search result.")
        return MOCK_RESULT

    api_key = os.getenv("SERPAPI_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "SERPAPI_KEY is not set. Add it to your .env file or pass --mock for offline mode."
        )

    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError("Image not found: {}".format(path))

    # SerpAPI Google Lens requires a public URL (not a direct file upload).
    # We temporarily host the image on a free public host first.
    image_url = _upload_image_for_url(path)

    log.info("Querying SerpAPI Google Lens with image URL ...")

    params = {
        "engine": "google_lens",
        "api_key": api_key,
        "url": image_url,
    }

    response = requests.get(
        SERPAPI_ENDPOINT,
        params=params,
        timeout=60,
    )

    if response.status_code != 200:
        raise WebSearchError(
            "SerpAPI returned HTTP {}: {}".format(response.status_code, response.text[:300])
        )

    data = response.json()

    # Parse visual matches
    raw_results = data.get("visual_matches") or data.get("organic_results") or []

    if not raw_results:
        raise WebSearchError(
            "SerpAPI returned no visual matches for the given image. "
            "Try a different face image or check your API key."
        )

    parsed = []
    for idx, item in enumerate(raw_results[:10], start=1):
        parsed.append(
            {
                "position": idx,
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "source": item.get("source", ""),
                "thumbnail": item.get("thumbnail", ""),
            }
        )

    best = parsed[0]

    log.info(
        "Search complete [OK]  %d results found. Best match: '%s' @ %s",
        len(parsed),
        best["title"][:60],
        best["link"][:80],
    )

    return {
        "engine": "google_lens",
        "search_url": data.get("search_metadata", {}).get("google_lens_url", ""),
        "results": parsed,
        "best_match": best,
    }


def extract_post_metadata(search_result: dict) -> dict:
    """
    Build a clean, fingerprint-ready metadata dict from the search result.

    This is the dict that will be hashed and anchored to the blockchain.
    """
    best = search_result.get("best_match", {})
    return {
        "source": best.get("source", "unknown"),
        "title": best.get("title", ""),
        "url": best.get("link", ""),
        "thumbnail": best.get("thumbnail", ""),
        "engine": search_result.get("engine", "google_lens"),
        "search_url": search_result.get("search_url", ""),
        "result_count": len(search_result.get("results", [])),
    }


# Quick self-test
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    img = sys.argv[1] if len(sys.argv) > 1 else "demo/sample_face.jpg"
    use_mock = "--mock" in sys.argv

    result = reverse_image_search(img, mock=use_mock)
    meta = extract_post_metadata(result)
    print("\n-- Search Result --")
    print(json.dumps(result, indent=2))
    print("\n-- Post Metadata (for blockchain) --")
    print(json.dumps(meta, indent=2))
