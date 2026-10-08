from pathlib import Path

import numpy as np

from bead_rl.core import BeadEnvironment
from bead_rl.ppo_legacy import PaperInspiredBeadEnvironment

ROOT = Path(__file__).resolve().parents[1]
SQUARE = ROOT / "data" / "benchmarks" / "training_square.jpeg"
CIRCLE = ROOT / "data" / "benchmarks" / "ood_circle.jpg"


def test_square_contour_extracts_and_is_bounded():
    env = BeadEnvironment(SQUARE, seed=42)
    assert env.path_points.ndim == 2
    assert env.path_points.shape[1] == 2
    assert 300 <= len(env.path_points) <= 400
    assert np.max(np.abs(env.path_points)) <= env.BOUNDARY + 1e-6


def test_fixed_seed_reset_is_reproducible():
    a = BeadEnvironment(CIRCLE, seed=123)
    b = BeadEnvironment(CIRCLE, seed=123)
    obs_a, _ = a.reset()
    obs_b, _ = b.reset()
    np.testing.assert_allclose(obs_a, obs_b, atol=0.0)


def test_sac_observation_and_step_contract():
    env = BeadEnvironment(SQUARE, seed=7)
    obs, _ = env.reset()
    assert obs.shape == (16,)
    next_obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert next_obs.shape == (16,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 0.0 <= info["coverage"] <= 1.0


def test_paper_observation_dimension_and_finite_reward():
    env = PaperInspiredBeadEnvironment(SQUARE, seed=7)
    obs, _ = env.reset()
    assert obs.shape == (25,)
    obs, reward, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == (25,)
    assert np.isfinite(reward)
