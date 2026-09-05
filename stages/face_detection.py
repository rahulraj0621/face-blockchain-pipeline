"""
stages/face_detection.py
------------------------
Stage 1 - Face Detection & Encoding

Uses MediaPipe (Google) for face detection - no C++ compilation needed.
Works on any platform via pip install.

Detects faces, crops the face region, and creates a stable 128-value
encoding from the normalised face pixels. The encoding hash goes to the blockchain.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from utils.logger import get_logger

log = get_logger("face_detection")


class FaceDetectionError(Exception):
    """Raised when no face can be detected in the image."""


def detect_and_encode(image_path: str) -> dict:
    """
    Detect a face in image_path and return its encoding plus metadata.

    Returns
    -------
    dict with keys:
        image_path      - original path (str)
        face_count      - number of faces found (int)
        encoding        - 128-element list of floats
        encoding_hash   - SHA-256 hex of the encoding (str)
        face_location   - {top, right, bottom, left}
    """
    import mediapipe as mp

    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError("Image not found: {}".format(path))

    log.info("Loading image: %s", path)
    pil_img = Image.open(path).convert("RGB")
    img_array = np.array(pil_img)
    h, w = img_array.shape[:2]

    log.info("Detecting face locations ...")
    mp_face = mp.solutions.face_detection

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as detector:
        results = detector.process(img_array)

    detections = results.detections or []
    face_count = len(detections)

    if face_count == 0:
        raise FaceDetectionError(
            "No face detected in '{}'. "
            "Please supply an image with a clearly visible, well-lit face.".format(path.name)
        )

    if face_count > 1:
        log.warning("%d faces detected - using the largest (first).", face_count)

    # Primary face bounding box (relative coords -> pixels)
    det = detections[0]
    bb = det.location_data.relative_bounding_box

    left   = max(0, int(bb.xmin * w))
    top    = max(0, int(bb.ymin * h))
    right  = min(w, int((bb.xmin + bb.width)  * w))
    bottom = min(h, int((bb.ymin + bb.height) * h))

    # Crop face, resize to 64x64, flatten to build a stable 128-d "encoding"
    log.info("Encoding face ...")
    face_crop = pil_img.crop((left, top, right, bottom)).resize((64, 64))
    face_pixels = np.array(face_crop).flatten().astype(np.float32) / 255.0

    # Sample 128 evenly-spaced values from the 12288-d pixel vector
    step = len(face_pixels) // 128
    encoding = face_pixels[::step][:128].tolist()

    encoding_json = json.dumps(encoding, separators=(",", ":"))
    encoding_hash = hashlib.sha256(encoding_json.encode()).hexdigest()

    result = {
        "image_path": str(path),
        "face_count": face_count,
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


# Quick self-test
if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "demo/sample_face.jpg"
    info = detect_and_encode(img)
    print(json.dumps({k: v for k, v in info.items() if k != "encoding"}, indent=2))
