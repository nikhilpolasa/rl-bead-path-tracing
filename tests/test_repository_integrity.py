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
    """Digest filenames + bytes in sorted order (same scheme as metadata)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_all_checkpoint_hashes_match_metadata():
    metadata = json.loads((ROOT / "models" / "checkpoint_metadata.json").read_text())
    for filename, record in metadata.items():
        path = ROOT / "models" / filename
        assert path.is_file(), filename
        assert sha256(path) == record["sha256"], filename


def test_dataset_digests_match_metadata():
    root = ROOT / "data" / "closed_shapes_dataset"
    metadata = json.loads((root / "DATASET_METADATA.json").read_text())
    for split in ("train", "val", "test"):
        split_dir = root / split
        files = [p for p in split_dir.iterdir() if p.is_file()]
        assert len(files) == metadata[split]["count"]
        assert dataset_digest(split_dir) == metadata[split]["aggregate_sha256"]


def test_documented_core_artifacts_exist():
    expected = [
        "README.md",
        "VERIFICATION.md",
        "results/experiment_registry.csv",
        "results/ood_circle_comparison_verified.json",
        "docs/reproducibility_audit.md",
        "docs/experiment_provenance.md",
        "docs/manuscript_corrections.md",
        "assets/sac_dagger_training_square.png",
        "assets/ood_model_comparison.png",
        "assets/ood_trajectory_comparison.png",
        "assets/procedural_batch_results.png",
    ]
    for relative in expected:
        assert (ROOT / relative).exists(), relative
