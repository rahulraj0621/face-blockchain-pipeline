"""
pipeline.py
────────────
Face Scan → Web Search → Blockchain Verification
End-to-end orchestrator.

Usage
─────
# Live mode (requires API keys in .env):
    python pipeline.py path/to/face.jpg

# Mock/offline mode (no API keys needed):
    python pipeline.py path/to/face.jpg --mock

# Use the bundled demo image:
    python pipeline.py demo/sample_face.jpg --mock
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from stages.face_detection import detect_and_encode, FaceDetectionError
from stages.web_search import reverse_image_search, extract_post_metadata, WebSearchError
from stages.blockchain import compute_fingerprint, store_on_chain, verify_on_chain
from utils.logger import get_logger

load_dotenv()
log = get_logger("pipeline")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   Face Scan → Web Search → Blockchain Verification Pipeline      ║
║   github.com/your-username/face-blockchain-pipeline              ║
╚══════════════════════════════════════════════════════════════════╝
"""

REPORT_PATH = Path("verification_report.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    log.info("━" * 60)
    log.info("  %s", title)
    log.info("━" * 60)


def _save_report(report: dict) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Report saved → %s", REPORT_PATH.resolve())


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(image_path: str, mock: bool = False) -> dict:
    """
    Execute the full 3-stage pipeline and return the verification report.

    Parameters
    ----------
    image_path : str
        Path to the input face image.
    mock : bool
        If True, use mock data for stages 2 and 3 (no external API calls).

    Returns
    -------
    dict – the complete verification report.
    """
    print(BANNER)
    started_at = datetime.now(timezone.utc).isoformat()

    report = {
        "started_at": started_at,
        "image_path": image_path,
        "mock_mode": mock,
        "stage_1_face_detection": {},
        "stage_2_web_search": {},
        "stage_3_blockchain": {},
        "verified": False,
        "finished_at": None,
    }

    # ── Stage 1: Face Detection ───────────────────────────────────────────────
    _section("STAGE 1 / 3  –  Face Detection & Encoding")
    try:
        face_info = detect_and_encode(image_path)
    except (FileNotFoundError, FaceDetectionError) as exc:
        log.error("Stage 1 failed: %s", exc)
        report["stage_1_face_detection"] = {"error": str(exc)}
        _save_report(report)
        return report

    report["stage_1_face_detection"] = {
        "face_count": face_info["face_count"],
        "encoding_hash": face_info["encoding_hash"],
        "face_location": face_info["face_location"],
    }
    log.info(
        "Stage 1 complete ✓  faces=%d  encoding_hash=%s…",
        face_info["face_count"],
        face_info["encoding_hash"][:16],
    )

    # ── Stage 2: Reverse Image Search ─────────────────────────────────────────
    _section("STAGE 2 / 3  –  Reverse Image Search (Google Lens)")
    try:
        search_result = reverse_image_search(image_path, mock=mock)
    except (WebSearchError, EnvironmentError, Exception) as exc:
        log.error("Stage 2 failed: %s", exc)
        report["stage_2_web_search"] = {"error": str(exc)}
        _save_report(report)
        return report

    post_meta = extract_post_metadata(search_result)

    report["stage_2_web_search"] = {
        "engine": search_result.get("engine"),
        "search_url": search_result.get("search_url"),
        "result_count": len(search_result.get("results", [])),
        "best_match": search_result.get("best_match"),
        "post_metadata": post_meta,
    }
    log.info(
        "Stage 2 complete ✓  %d results  best='%s'",
        len(search_result.get("results", [])),
        search_result["best_match"]["title"][:60],
    )

    # ── Stage 3: Blockchain Verification ──────────────────────────────────────
    _section("STAGE 3 / 3  –  Blockchain Fingerprint & Verification")

    # Build the data blob to fingerprint: face encoding hash + post metadata
    fingerprint_data = {
        "face_encoding_hash": face_info["encoding_hash"],
        "post": post_meta,
        "pipeline_version": "1.0.0",
        "timestamp": started_at,
    }

    fingerprint = compute_fingerprint(fingerprint_data)
    log.info("Fingerprint: %s", fingerprint)

    try:
        chain_result = store_on_chain(fingerprint, mock=mock)
    except (EnvironmentError, Exception) as exc:
        log.error("Stage 3 (store) failed: %s", exc)
        report["stage_3_blockchain"] = {"error": str(exc), "fingerprint": fingerprint}
        _save_report(report)
        return report

    log.info("Stored on chain ✓  tx_hash=%s", chain_result["tx_hash"])
    if chain_result.get("etherscan_url"):
        log.info("Etherscan: %s", chain_result["etherscan_url"])

    # Re-verify
    verified = verify_on_chain(chain_result["tx_hash"], fingerprint, mock=mock)

    report["stage_3_blockchain"] = {
        "fingerprint": fingerprint,
        "fingerprint_data": fingerprint_data,
        "tx_hash": chain_result["tx_hash"],
        "etherscan_url": chain_result.get("etherscan_url", ""),
        "mock": chain_result.get("mock", False),
        "verified": verified,
    }
    report["verified"] = verified
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    _save_report(report)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("━" * 60)
    log.info("  PIPELINE COMPLETE")
    log.info("━" * 60)
    log.info("  Face encoding hash : %s…", face_info["encoding_hash"][:32])
    log.info("  Best match         : %s", search_result["best_match"]["title"][:60])
    log.info("  Match URL          : %s", search_result["best_match"]["link"])
    log.info("  Fingerprint        : %s…", fingerprint[:40])
    log.info("  Transaction hash   : %s", chain_result["tx_hash"])
    if chain_result.get("etherscan_url"):
        log.info("  Etherscan          : %s", chain_result["etherscan_url"])
    log.info(
        "  Verification       : %s",
        "PASSED ✓" if verified else "FAILED ✗",
    )
    log.info("━" * 60)

    return report


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face Scan → Web Search → Blockchain Verification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="demo/sample_face.jpg",
        help="Path to the input face image (default: demo/sample_face.jpg)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/offline mode – no real API calls or ETH transactions",
    )
    parser.add_argument(
        "--output",
        default="verification_report.json",
        help="Path for the JSON verification report (default: verification_report.json)",
    )
    args = parser.parse_args()

    global REPORT_PATH
    REPORT_PATH = Path(args.output)

    report = run_pipeline(args.image, mock=args.mock)

    # Exit with non-zero code if verification failed
    sys.exit(0 if report.get("verified") else 1)


if __name__ == "__main__":
    main()
