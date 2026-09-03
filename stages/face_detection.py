"""
stages/face_detection.py
------------------------
Stage 1 - Face Detection & Encoding

Given a path to an image:
  - Detects faces using face_recognition (dlib under the hood).
  - Returns the 128-dimensional face encoding.
  - Also returns a SHA-256 hash of that encoding for downstream use.

Dependencies:  face_recognition, numpy, Pillow
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import face_recognition
import numpy as np
from PIL import Image

from utils.logger import get_logger

log = get_logger("face_detection")


class FaceDetectionError(Exception):
    """Raised when no face (or multiple ambiguous faces) can be detected."""


def detect_and_encode(image_path: str) -> dict:
    """
    Detect a face in image_path and return its 128-d encoding plus metadata.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the input image (JPG / PNG / etc.).

    Returns
    -------
    dict with keys:
        image_path      - original path (str)
        face_count      - number of faces found (int)
        encoding        - 128-element list of floats
        encoding_hash   - SHA-256 hex of the JSON-serialised encoding (str)
        face_location   - (top, right, bottom, left) of the primary face
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError("Image not found: {}".format(path))

    log.info("Loading image: %s", path)

    # Load via PIL first to normalise format, then convert to RGB numpy array
    pil_img = Image.open(path).convert("RGB")
    img_array = np.array(pil_img)

    log.info("Detecting face locations ...")
    locations = face_recognition.face_locations(img_array, model="hog")
    face_count = len(locations)

    if face_count == 0:
        raise FaceDetectionError(
            "No face detected in '{}'. "
            "Please supply an image with a clearly visible face.".format(path.name)
        )

    if face_count > 1:
        log.warning(
            "%d faces detected - using the largest (first) face for encoding.",
            face_count,
        )

    log.info("Encoding face ...")
    encodings = face_recognition.face_encodings(img_array, known_face_locations=locations)
    if not encodings:
        raise FaceDetectionError("Face location found but encoding failed.")

    primary_encoding: np.ndarray = encodings[0]
    primary_location: tuple = locations[0]

    # Serialise encoding to a stable JSON string for hashing
    encoding_list = primary_encoding.tolist()
    encoding_json = json.dumps(encoding_list, separators=(",", ":"), sort_keys=False)
    encoding_hash = hashlib.sha256(encoding_json.encode()).hexdigest()

    result = {
        "image_path": str(path),
        "face_count": face_count,
        "encoding": encoding_list,
        "encoding_hash": encoding_hash,
        "face_location": {
            "top": primary_location[0],
            "right": primary_location[1],
            "bottom": primary_location[2],
            "left": primary_location[3],
        },
    }

    log.info(
        "Face detected [OK]  location=%s  encoding_hash=%s...",
        primary_location,
        encoding_hash[:16],
    )
    return result


# Quick self-test
if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else "demo/sample_face.jpg"
    info = detect_and_encode(img)
    print(json.dumps({k: v for k, v in info.items() if k != "encoding"}, indent=2))
