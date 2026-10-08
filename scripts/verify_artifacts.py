#!/usr/bin/env python
"""Verify hashes/counts of preserved public experiment artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    model_meta = json.loads((ROOT / "models/checkpoint_metadata.json").read_text())
    for filename, record in model_meta.items():
        actual = sha256(ROOT / "models" / filename)
        if actual != record["sha256"]:
            raise SystemExit(f"checkpoint hash mismatch: {filename}")

    data_root = ROOT / "data/closed_shapes_dataset"
    data_meta = json.loads((data_root / "DATASET_METADATA.json").read_text())
    for split, record in data_meta.items():
        directory = data_root / split
        files = [path for path in directory.iterdir() if path.is_file()]
        if len(files) != record["count"]:
            raise SystemExit(f"dataset count mismatch: {split}")
        if dataset_digest(directory) != record["aggregate_sha256"]:
            raise SystemExit(f"dataset digest mismatch: {split}")

    print(
        f"verified {len(model_meta)} checkpoint hashes and "
        f"{sum(record['count'] for record in data_meta.values())} dataset images"
    )


if __name__ == "__main__":
    main()
