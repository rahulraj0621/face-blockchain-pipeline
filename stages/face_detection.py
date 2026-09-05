"""
stages/face_detection.py
------------------------
Stage 1 - Face Detection & Encoding

Uses OpenCV Haar Cascades — built into opencv-python-headless.
No downloads, no compilation, no external APIs needed.
Works identically on Windows, Mac, Linux, Railway, Docker.
"""

import cv2
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from utils.logger import get_logger

log = get_logger("face_detection")

# Bundled haarcascade XML - committed to the repo, works on all OpenCV versions
HAAR_PATH = str(Path(__file__).parent / "haarcascade_frontalface_default.xml")


class FaceDetectionError(Exception):
    """Raised when no face can be detected in the image."""


def detect_and_encode(image_path: str) -> dict:
    """
    Detect a face in image_path and return its encoding plus metadata.

    Returns dict with keys:
        image_path      - original path (str)
        face_count      - number of faces found (int)
        encoding        - 128-element list of floats
        encoding_hash   - SHA-256 hex of the encoding (str)
        face_location   - {top, right, bottom, left}
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError("Image not found: {}".format(path))

    log.info("Loading image: %s", path)

    # Read with OpenCV
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        # Fallback: try PIL (handles PNGs etc. better on some systems)
        pil_img = Image.open(path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    log.info("Detecting face locations ...")

    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    face_count = len(faces) if hasattr(faces, "__len__") else 0

    if face_count == 0:
        raise FaceDetectionError(
            "No face detected in '{}'. "
            "Please use a clear, front-facing, well-lit photo.".format(path.name)
        )

    if face_count > 1:
        log.warning("%d faces detected - using the largest.", face_count)
        # Sort by area descending, take largest
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    x, y, fw, fh = faces[0]
    left, top, right, bottom = int(x), int(y), int(x + fw), int(y + fh)

    # Crop face, normalise to 64x64, build 128-d encoding vector
    log.info("Encoding face ...")
    face_crop = pil_img.crop((left, top, right, bottom)).resize((64, 64))
    face_pixels = np.array(face_crop).flatten().astype(np.float32) / 255.0

    # Sample 128 evenly-spaced values from the 12288-d pixel vector
    step = max(1, len(face_pixels) // 128)
    encoding = face_pixels[::step][:128].tolist()

    encoding_json = json.dumps(encoding, separators=(",", ":"))
    encoding_hash = hashlib.sha256(encoding_json.encode()).hexdigest()

    result = {
        "image_path": str(path),
        "face_count": int(face_count),
        "encoding": encoding,
        "encoding_hash": encoding_hash,
        "face_location": {
            "top": top, "right": right, "bottom": bottom, "left": left,
        },
    }

    log.info(
        "Face detected [OK]  location=(%d,%d,%d,%d)  encoding_hash=%s...",
        top, right, bottom, left, encoding_hash[:16],
    )
    return result


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "demo/sample_face.jpg"
    info = detect_and_encode(img)
    print(json.dumps({k: v for k, v in info.items() if k != "encoding"}, indent=2))
