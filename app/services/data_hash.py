"""Tracks a hash of the data directory to detect changes between restarts."""
import hashlib
import json
from pathlib import Path


def compute_data_hash(data_dir: Path) -> str:
    """Return a SHA-256 hash over all JSON filenames + their sizes + mtimes."""
    hasher = hashlib.sha256()
    for f in sorted(data_dir.glob("*.json")):
        stat = f.stat()
        hasher.update(f.name.encode())
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime).encode())
    return hasher.hexdigest()


def read_stored_hash(hash_file: Path) -> str | None:
    if not hash_file.exists():
        return None
    try:
        return json.loads(hash_file.read_text())["hash"]
    except Exception:
        return None


def write_stored_hash(hash_file: Path, hash_value: str) -> None:
    hash_file.write_text(json.dumps({"hash": hash_value}))
