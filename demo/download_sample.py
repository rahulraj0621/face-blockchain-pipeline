"""
demo/download_sample.py
────────────────────────
Downloads a public-domain face image for use as the demo input.

Sources tried in order (all public domain / CC0):
  1. thispersondoesnotexist.com  – AI-generated faces, no copyright
  2. Wikimedia Commons portrait  – CC0 public domain photo

Run:
    python demo/download_sample.py
"""

import urllib.request
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "sample_face.jpg"

# Public-domain portrait images (direct full-resolution URLs)
# All sourced from Wikimedia Commons
SOURCES = [
    # Albert Einstein - public domain
    "https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg",
    # Abraham Lincoln - public domain
    "https://upload.wikimedia.org/wikipedia/commons/a/ab/Abraham_Lincoln_O-77_by_Gardner%2C_1863.jpg",
    # Marie Curie - public domain
    "https://upload.wikimedia.org/wikipedia/commons/c/c8/Marie_Curie_c._1920s.jpg",
    # Thomas Edison - public domain
    "https://upload.wikimedia.org/wikipedia/commons/9/9d/Thomas_Edison2.jpg",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/jpeg,image/*;q=0.9,*/*;q=0.8",
}





def download_sample():
    print("Downloading sample face image to {} ...".format(OUTPUT_PATH))
    last_err = None
    for url in SOURCES:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if len(data) < 1000:
                print("  Response too small ({} bytes), trying next source ...".format(len(data)))
                continue
            OUTPUT_PATH.write_bytes(data)
            print("Saved {:,} bytes -> {}".format(len(data), OUTPUT_PATH))
            return
        except Exception as exc:
            print("  Failed ({}...): {}".format(url[:60], exc))
            last_err = exc

    raise RuntimeError(
        "All download sources failed. Last error: {}\n"
        "Please manually place a face photo at demo/sample_face.jpg".format(last_err)
    )


if __name__ == "__main__":
    download_sample()
