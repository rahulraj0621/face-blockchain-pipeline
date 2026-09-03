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


def _resize_image_for_upload(image_path) -> bytes:
    """
    Resize image to max 800px on any side and return JPEG bytes.
    Reduces a 2-3MB photo down to ~50-100KB for fast, reliable upload.
    """
    from PIL import Image
    import io

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((800, 800), Image.LANCZOS)  # shrink in-place, keep aspect ratio

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    size_kb = len(buf.getvalue()) // 1024
    log.info("Image resized to %dx%d (%d KB) for upload.", img.width, img.height, size_kb)
    return buf.getvalue()


def _upload_image_for_url(image_path) -> str:
    """
    Upload a local image to a free temporary host and return its public URL.

    Resizes the image first (to ~800px max), then tries multiple free hosts
    in order: catbox.moe -> uguu.se -> 0x0.st -> file.io
    """
    img_bytes = _resize_image_for_upload(image_path)
    fname = "face_search.jpg"

    # --- Host 1: catbox.moe (very reliable, permanent, no auth) ---
    try:
        log.info("Hosting image on catbox.moe ...")
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (fname, img_bytes, "image/jpeg")},
            timeout=30,
        )
        if resp.status_code == 200 and resp.text.strip().startswith("http"):
            url = resp.text.strip()
            log.info("Image hosted at: %s", url)
            return url
        log.warning("catbox.moe response (%d): %s", resp.status_code, resp.text[:80])
    except Exception as exc:
        log.warning("catbox.moe failed: %s", exc)

    # --- Host 2: uguu.se (temporary, 48h, no auth) ---
    try:
        log.info("Hosting image on uguu.se ...")
        resp = requests.post(
            "https://uguu.se/upload",
            files={"files[]": (fname, img_bytes, "image/jpeg")},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("files", [])
            if files and files[0].get("url"):
                url = files[0]["url"]
                log.info("Image hosted at: %s", url)
                return url
        log.warning("uguu.se response (%d): %s", resp.status_code, resp.text[:80])
    except Exception as exc:
        log.warning("uguu.se failed: %s", exc)

    # --- Host 3: 0x0.st ---
    try:
        log.info("Hosting image on 0x0.st ...")
        resp = requests.post(
            "https://0x0.st",
            files={"file": (fname, img_bytes, "image/jpeg")},
            timeout=30,
        )
        if resp.status_code == 200 and resp.text.strip().startswith("http"):
            url = resp.text.strip()
            log.info("Image hosted at: %s", url)
            return url
        log.warning("0x0.st response (%d): %s", resp.status_code, resp.text[:80])
    except Exception as exc:
        log.warning("0x0.st failed: %s", exc)

    # --- Host 4: file.io ---
    try:
        log.info("Hosting image on file.io ...")
        resp = requests.post(
            "https://file.io/?expires=1h",
            files={"file": (fname, img_bytes, "image/jpeg")},
            timeout=30,
        )
        if resp.status_code in (200, 201):
            try:
                data = resp.json()
                url = data.get("link") or data.get("url", "")
                if url:
                    log.info("Image hosted at: %s", url)
                    return url
            except Exception:
                pass
        log.warning("file.io response (%d): %s", resp.status_code, resp.text[:80])
    except Exception as exc:
        log.warning("file.io failed: %s", exc)

    raise WebSearchError(
        "All image hosts failed. Check your internet connection, "
        "or run with --mock flag for offline mode."
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
