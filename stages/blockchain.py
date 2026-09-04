"""
stages/blockchain.py
--------------------
Stage 3 - Blockchain Verification

Records a SHA-256 fingerprint of the discovered social-media post on the
Ethereum Sepolia testnet, then re-verifies it by fetching the transaction
and comparing its input data to the expected fingerprint.

Environment variables required (in .env):
    INFURA_PROJECT_ID  - Infura Web3 project ID
    ETH_PRIVATE_KEY    - private key of a Sepolia-funded wallet (0x...)
    ETH_ADDRESS        - corresponding public address (0x...)

Mock mode (--mock flag or MOCK_MODE=1):
    Simulates the full flow in-memory - no real transaction is sent.
"""

import hashlib
import json
import os
from typing import Optional

from utils.logger import get_logger

log = get_logger("blockchain")


# Constants
SEPOLIA_CHAIN_ID = 11155111
ETHERSCAN_BASE = "https://sepolia.etherscan.io/tx/"


# Fingerprint helpers

def compute_fingerprint(data: dict) -> str:
    """
    Return a deterministic SHA-256 fingerprint of data.

    The dict is serialised to JSON with sorted keys so the hash is stable
    regardless of insertion order.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    log.info("Fingerprint computed: sha256:%s...", digest[:16])
    return "sha256:{}".format(digest)


def _fingerprint_to_hex(fingerprint: str) -> str:
    """Convert 'sha256:<hex>' to a 0x-prefixed hex string for on-chain storage."""
    raw = fingerprint.replace("sha256:", "")
    return "0x" + raw


# Mock blockchain

class _MockChain:
    """In-memory simulated blockchain for demo / offline use."""

    _store: dict = {}

    @classmethod
    def send(cls, fingerprint: str) -> str:
        """'Broadcast' a transaction and return a fake tx hash."""
        fake_tx = "0xMOCK_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:40]
        cls._store[fake_tx] = fingerprint
        log.info("[MOCK] Transaction stored: %s", fake_tx)
        return fake_tx

    @classmethod
    def fetch(cls, tx_hash: str) -> Optional[str]:
        return cls._store.get(tx_hash)


# Real Ethereum helpers

def _get_web3():
    """Build a Web3 connection to Sepolia via Infura."""
    try:
        from web3 import Web3
    except ImportError:
        raise ImportError("web3 package not installed. Run: pip install web3")

    project_id = os.getenv("INFURA_PROJECT_ID", "")
    if not project_id:
        raise EnvironmentError(
            "INFURA_PROJECT_ID not set. Add it to your .env file."
        )

    rpc_url = "https://sepolia.infura.io/v3/{}".format(project_id)
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    if not w3.is_connected():
        raise ConnectionError(
            "Cannot connect to Ethereum Sepolia via Infura (RPC: {}). "
            "Check your INFURA_PROJECT_ID and network connection.".format(rpc_url)
        )

    log.info("Connected to Ethereum Sepolia  (chainId=%d)", w3.eth.chain_id)
    return w3


def _send_transaction(fingerprint: str, w3, private_key: str, address: str) -> str:
    """
    Send a zero-value Ethereum transaction whose input field carries the
    fingerprint.  Returns the transaction hash.
    """
    hex_data = _fingerprint_to_hex(fingerprint)

    nonce = w3.eth.get_transaction_count(address)
    gas_price = w3.eth.gas_price

    tx = {
        "nonce": nonce,
        "to": address,          # self-send; keeps it simple and cheap
        "value": 0,
        "gas": 50_000,
        "gasPrice": gas_price,
        "chainId": SEPOLIA_CHAIN_ID,
        "data": hex_data,
    }

    log.info("Signing transaction ...")
    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)

    log.info("Broadcasting transaction to Sepolia ...")
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    tx_hash_hex = tx_hash.hex()

    log.info("Transaction broadcast [OK]  tx_hash=%s", tx_hash_hex)
    log.info("Waiting for receipt (up to 300 s) ...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
    if receipt["status"] != 1:
        raise RuntimeError("Transaction failed on-chain: {}".format(receipt))

    log.info("Transaction mined [OK]  block=%d", receipt["blockNumber"])
    return tx_hash_hex


def _verify_transaction(tx_hash: str, fingerprint: str, w3) -> bool:
    """
    Fetch the transaction from the chain and confirm its input data matches
    the expected fingerprint.
    """
    log.info("Fetching transaction for verification ...")
    tx = w3.eth.get_transaction(tx_hash)
    on_chain_data: str = tx["input"].hex()

    expected = _fingerprint_to_hex(fingerprint)
    # Normalise: strip 0x and compare lowercase
    match = on_chain_data.lower().lstrip("0x") == expected.lower().lstrip("0x")

    if match:
        log.info("Verification PASSED [OK] - on-chain data matches fingerprint.")
    else:
        log.error(
            "Verification FAILED [X]\n  expected: %s\n  on-chain: %s",
            expected,
            on_chain_data,
        )
    return match


# Public API

def store_on_chain(fingerprint: str, mock: bool = False) -> dict:
    """
    Store fingerprint on-chain and return a result dict.

    Parameters
    ----------
    fingerprint : str
        Output of compute_fingerprint() - e.g. 'sha256:abc123...'
    mock : bool
        If True, use the in-memory mock chain (no real ETH spent).

    Returns
    -------
    dict with keys:
        tx_hash         - transaction hash string
        etherscan_url   - link to Etherscan (empty in mock mode)
        fingerprint     - the fingerprint that was stored
        mock            - bool indicating mock mode
    """
    if mock or os.getenv("MOCK_MODE", "0") == "1":
        log.info("[MOCK] Storing fingerprint on mock chain ...")
        tx_hash = _MockChain.send(fingerprint)
        return {
            "tx_hash": tx_hash,
            "etherscan_url": "",
            "fingerprint": fingerprint,
            "mock": True,
        }

    private_key = os.getenv("ETH_PRIVATE_KEY", "")
    address = os.getenv("ETH_ADDRESS", "")
    if not private_key or not address:
        raise EnvironmentError(
            "ETH_PRIVATE_KEY and ETH_ADDRESS must be set in your .env file. "
            "Use --mock for offline mode."
        )

    w3 = _get_web3()
    tx_hash = _send_transaction(fingerprint, w3, private_key, address)

    return {
        "tx_hash": tx_hash,
        "etherscan_url": ETHERSCAN_BASE + tx_hash,
        "fingerprint": fingerprint,
        "mock": False,
    }


def verify_on_chain(tx_hash: str, fingerprint: str, mock: bool = False) -> bool:
    """
    Re-verify that fingerprint was recorded in the transaction tx_hash.

    Parameters
    ----------
    tx_hash : str
        Transaction hash returned by store_on_chain().
    fingerprint : str
        The fingerprint to verify against.
    mock : bool
        If True, verify against the in-memory mock chain.

    Returns
    -------
    bool - True if the on-chain data matches the fingerprint.
    """
    if mock or os.getenv("MOCK_MODE", "0") == "1":
        log.info("[MOCK] Verifying fingerprint against mock chain ...")
        stored = _MockChain.fetch(tx_hash)
        match = stored == fingerprint
        if match:
            log.info("[MOCK] Verification PASSED [OK]")
        else:
            log.error("[MOCK] Verification FAILED [X]  stored=%s", stored)
        return match

    w3 = _get_web3()
    return _verify_transaction(tx_hash, fingerprint, w3)


# Quick self-test
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    use_mock = "--mock" in sys.argv

    sample = {
        "title": "Test post",
        "url": "https://example.com/post/1",
        "source": "Example",
    }

    fp = compute_fingerprint(sample)
    print("\nFingerprint: {}".format(fp))

    result = store_on_chain(fp, mock=use_mock)
    print("\nStore result:\n{}".format(json.dumps(result, indent=2)))

    verified = verify_on_chain(result["tx_hash"], fp, mock=use_mock)
    print("\nVerification: {}".format("PASSED [OK]" if verified else "FAILED [X]"))
