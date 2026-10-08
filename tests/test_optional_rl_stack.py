"""Tests that run in CI/full installs but skip in minimal audit environments."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None or importlib.util.find_spec("stable_baselines3") is None,
    reason="Gymnasium/Stable-Baselines3 not installed in this environment",
)


def test_gym_environment_contract():
    from stable_baselines3.common.env_checker import check_env
    from bead_rl.gym_envs import BeadTraceEnv, PaperTraceEnv

    image = ROOT / "data" / "benchmarks" / "training_square.jpeg"
    check_env(BeadTraceEnv(image, seed=1), warn=True)
    check_env(PaperTraceEnv(image, seed=1), warn=True)


def test_archived_ppo_checkpoints_load_with_compatibility_shim():
    from bead_rl.policies import load_archived_ppo

    for name in ("ppo_single_final.zip", "ppo_multishape_best.zip"):
        model = load_archived_ppo(ROOT / "models" / name, device="cpu")
        assert tuple(model.observation_space.shape) == (25,)
        assert tuple(model.action_space.shape) == (2,)
