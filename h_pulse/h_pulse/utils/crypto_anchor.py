from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

try:
    from web3 import Web3
    from web3.providers.eth_tester import EthereumTesterProvider  # type: ignore
    _HAS_WEB3 = True
except Exception:  # pragma: no cover
    Web3 = None  # type: ignore
    EthereumTesterProvider = None  # type: ignore
    _HAS_WEB3 = False


@dataclass
class AnchorResult:
    sha3_256_hex: str
    signature_hex: str
    public_key_hex: str
    tx_hash: str
    block_number: int
    synthetic: bool
    timestamp: str


def sha3_256_hex(data: bytes) -> str:
    digest = hashes.Hash(hashes.SHA3_256())
    digest.update(data)
    return digest.finalize().hex()


def generate_ed25519_keypair() -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def export_keys_to_hex(private_key: Ed25519PrivateKey, public_key: Ed25519PublicKey) -> Tuple[str, str]:
    priv = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()).hex()
    pub = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return priv, pub


def sign_message(private_key: Ed25519PrivateKey, message: bytes) -> bytes:
    return private_key.sign(message)


def verify_signature(public_key: Ed25519PublicKey, message: bytes, signature: bytes) -> bool:
    try:
        public_key.verify(signature, message)
        return True
    except Exception:
        return False


def anchor_with_web3_or_synthetic(payload: dict, private_key: Optional[Ed25519PrivateKey] = None) -> AnchorResult:
    message = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest_hex = sha3_256_hex(message)

    if private_key is None:
        private_key, _ = generate_ed25519_keypair()
    public_key = private_key.public_key()

    signature = sign_message(private_key, bytes.fromhex(digest_hex))

    timestamp = datetime.now(timezone.utc).isoformat()

    # Try real local in-memory chain via eth-tester if available
    if _HAS_WEB3 and EthereumTesterProvider is not None:
        try:
            w3 = Web3(EthereumTesterProvider())
            # Deploy a no-op tx by sending to self with data as digest
            acct = w3.eth.accounts[0]
            tx = {
                "from": acct,
                "to": acct,
                "value": 0,
                "data": bytes.fromhex(digest_hex),
            }
            tx_hash = w3.eth.send_transaction(tx)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            return AnchorResult(
                sha3_256_hex=digest_hex,
                signature_hex=signature.hex(),
                public_key_hex=public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
                tx_hash=tx_hash.hex(),
                block_number=receipt["blockNumber"],
                synthetic=False,
                timestamp=timestamp,
            )
        except Exception:
            pass

    # Fallback synthetic anchor: derive tx_hash-like value locally
    synthetic_tx = sha3_256_hex(b"H-Pulse::synthetic::" + bytes.fromhex(digest_hex))
    return AnchorResult(
        sha3_256_hex=digest_hex,
        signature_hex=signature.hex(),
        public_key_hex=public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
        tx_hash="0x" + synthetic_tx,
        block_number=0,
        synthetic=True,
        timestamp=timestamp,
    )


def verify_anchor(digest_hex: str, signature_hex: str, public_key_hex: str) -> bool:
    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
    return verify_signature(pub, bytes.fromhex(digest_hex), bytes.fromhex(signature_hex))