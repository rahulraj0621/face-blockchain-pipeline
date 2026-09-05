"""
app.py - Flask Web UI
Face Scan -> Web Search -> Blockchain Verification
"""

import os
import uuid
import json
import threading
import io
import time
import numpy as np
import requests as http_requests

from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

os.makedirs("uploads", exist_ok=True)

# ---------- In-memory job store ----------
_jobs = {}
_lock = threading.Lock()


def _set(job_id, **kw):
    with _lock:
        _jobs[job_id].update(kw)


def _log(job_id, stage, msg, status="running"):
    with _lock:
        _jobs[job_id]["logs"].append({
            "stage": stage,
            "msg": msg,
            "status": status,
            "t": datetime.now().strftime("%H:%M:%S"),
        })


# ---------- Face similarity helper ----------
def compute_similarity(input_encoding_list, thumbnail_url):
    """Download thumbnail, detect face with OpenCV, return cosine similarity %."""
    import cv2
    try:
        resp = http_requests.get(
            thumbnail_url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if resp.status_code != 200:
            return 0.0

        pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        haar = str(Path(__file__).parent / "stages" / "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(haar)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )

        if not hasattr(faces, "__len__") or len(faces) == 0:
            return 0.0

        x, y, fw, fh = faces[0]
        face_crop = pil.crop((x, y, x + fw, y + fh)).resize((64, 64))
        face_pixels = np.array(face_crop).flatten().astype(np.float32) / 255.0
        step = max(1, len(face_pixels) // 128)
        match_enc = face_pixels[::step][:128]

        inp = np.array(input_encoding_list[:128])
        dot = float(np.dot(inp, match_enc))
        norm = float(np.linalg.norm(inp) * np.linalg.norm(match_enc))
        if norm == 0:
            return 0.0
        return round(max(0.0, (dot / norm) * 100.0), 1)
    except Exception:
        return 0.0




# ---------- Background pipeline runner ----------
def run_job(job_id, image_path, mock):
    try:
        # STAGE 1 - Face Detection
        _log(job_id, "1", "Loading image and detecting face...")
        from stages.face_detection import detect_and_encode, FaceDetectionError
        try:
            fi = detect_and_encode(image_path)
        except FaceDetectionError as e:
            _set(job_id, status="error", error=str(e))
            return
        except FileNotFoundError as e:
            _set(job_id, status="error", error=str(e))
            return

        _log(job_id, "1",
             "{} face(s) found. Hash: {}...".format(fi["face_count"], fi["encoding_hash"][:20]),
             "done")
        _set(job_id, face_info={
            "face_count": fi["face_count"],
            "encoding_hash": fi["encoding_hash"],
            "face_location": fi["face_location"],
        })

        # STAGE 2 - Web Search
        _log(job_id, "2", "Hosting image and querying Google Lens...")
        from stages.web_search import reverse_image_search, extract_post_metadata, WebSearchError
        try:
            sr = reverse_image_search(image_path, mock=mock)
        except Exception as e:
            _set(job_id, status="error", error="Web search: {}".format(e))
            return

        meta = extract_post_metadata(sr)
        best = sr.get("best_match", {})
        _log(job_id, "2",
             "Found {} results. Best: {}".format(
                 sr.get("result_count", 0), best.get("title", "")[:60]),
             "done")

        # Compute similarity
        _log(job_id, "2", "Computing neural face similarity score...")
        sim = 0.0
        thumb = best.get("thumbnail", "")
        if thumb:
            sim = compute_similarity(fi["encoding"], thumb)
        _log(job_id, "2", "Neural similarity: {}%".format(sim), "done")
        _set(job_id, search=sr, meta=meta, similarity=sim)

        # STAGE 3 - Blockchain
        _log(job_id, "3", "Computing SHA-256 cryptographic fingerprint...")
        from stages.blockchain import compute_fingerprint, store_on_chain, verify_on_chain

        fp_data = {
            "face_encoding_hash": fi["encoding_hash"],
            "post": meta,
            "pipeline_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        fp = compute_fingerprint(fp_data)
        _log(job_id, "3", "Fingerprint: {}...".format(fp[:36]))

        _log(job_id, "3", "Broadcasting transaction to Ethereum Sepolia...")
        try:
            chain = store_on_chain(fp, mock=mock)
        except Exception as e:
            _set(job_id, status="error", error="Blockchain: {}".format(e))
            return

        _log(job_id, "3",
             "Mined in block! TX: {}...".format(chain["tx_hash"][:24]),
             "done")

        verified = verify_on_chain(chain["tx_hash"], fp, mock=mock)
        _log(job_id, "3",
             "Ledger verification: {}".format("PASSED" if verified else "FAILED"),
             "done" if verified else "error")

        _set(job_id,
             status="done",
             fingerprint=fp,
             chain=chain,
             verified=verified)

    except Exception as e:
        _set(job_id, status="error", error=str(e))


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400
    f = request.files["photo"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    mock = request.form.get("mock", "false").lower() == "true"

    job_id = str(uuid.uuid4())
    fname = job_id + "_" + secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    f.save(path)

    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "logs": [],
            "face_info": None,
            "search": None,
            "meta": None,
            "similarity": 0.0,
            "fingerprint": None,
            "chain": None,
            "verified": False,
            "error": None,
        }

    t = threading.Thread(target=run_job, args=(job_id, path, mock), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
