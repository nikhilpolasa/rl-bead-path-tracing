import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_data(name: str):
    with zipfile.ZipFile(ROOT / "models" / name) as archive:
        return json.loads(archive.read("data")), archive.read("_stable_baselines3_version").decode().strip()


def test_final_sac_metadata_matches_archived_model():
    data, version = checkpoint_data("sac_dagger_final.zip")
    assert version == "2.9.0"
    assert data["observation_space"]["_shape"] == [16]
    assert data["action_space"]["_shape"] == [2]
    assert data["num_timesteps"] == 30000
    assert data["gamma"] == 0.995


def test_multishape_best_is_5_4m_checkpoint():
    data, version = checkpoint_data("ppo_multishape_best.zip")
    assert version == "2.9.0"
    assert data["observation_space"]["_shape"] == [25]
    assert data["num_timesteps"] == 5_400_004
    assert data["learning_rate"] == 5e-05


def test_dataset_split_counts_are_exact():
    metadata = json.loads((ROOT / "data" / "closed_shapes_dataset" / "DATASET_METADATA.json").read_text())
    assert metadata["train"]["count"] == 700
    assert metadata["val"]["count"] == 150
    assert metadata["test"]["count"] == 150
