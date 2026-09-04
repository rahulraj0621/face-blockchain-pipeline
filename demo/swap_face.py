"""
Swaps demo/sample_face.jpg with Rohit Sharma's photo for testing.
Run: python demo/swap_face.py
"""
import urllib.request
import shutil
import os
from pathlib import Path

OUTPUT = Path(__file__).parent / "sample_face.jpg"

# Rohit Sharma - Wikimedia Commons (public domain / freely licensed)
SOURCES = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Rohit_Sharma_official_image.jpg/440px-Rohit_Sharma_official_image.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/5b/Rohit_Sharma_official_image.jpg",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "image/jpeg,image/*",
}

def swap():
    print("Downloading Rohit Sharma photo ...")
    for url in SOURCES:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if len(data) > 5000:
                OUTPUT.write_bytes(data)
                print("Downloaded: {:,} bytes -> {}".format(len(data), OUTPUT))
                return
            else:
                print("Too small ({} bytes), trying next ...".format(len(data)))
        except Exception as e:
            print("Failed ({}): {}".format(url[:60], e))
    print("All sources failed.")

if __name__ == "__main__":
    swap()
