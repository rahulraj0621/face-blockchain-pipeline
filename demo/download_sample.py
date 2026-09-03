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

# Sources in priority order - tries each until one succeeds
SOURCES = [
    # AI-generated face (thispersondoesnotexist) - refreshes on each request
    "https://thispersondoesnotexist.com/",
    # Fallback: Wikimedia Commons public-domain portraits (correct sizes)
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/"
        "Albert_Einstein_1947.jpg/220px-Albert_Einstein_1947.jpg"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/"
        "Abraham_Lincoln_O-77_matte_collodion_print.jpg/"
        "220px-Abraham_Lincoln_O-77_matte_collodion_print.jpg"
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/"
        "Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/"
        "402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
    ),
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
