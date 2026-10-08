"""Gymnasium wrappers used by Stable-Baselines3 training scripts."""
from __future__ import annotations

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import BeadEnvironment
from .ppo_legacy import PaperInspiredBeadEnvironment


class BeadTraceEnv(gym.Env):
    """16-D continuous environment used by the SAC + DAgger pipeline."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        image_path: str | Path | None = None,
        seed: int = 42,
        success_coverage: float = 0.99,
    ) -> None:
        super().__init__()
        self._env = BeadEnvironment(
            image_path=image_path,
            seed=seed,
            success_coverage=success_coverage,
        )
        self._image_path = image_path
        self._seed = seed
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(BeadEnvironment.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        # Preserve the archived wrapper semantics: Gymnasium receives ``seed``
        # but the core environment continues from the RNG seeded at construction.
        # Historical training/evaluation code seeded each core env explicitly in
        # its constructor rather than reseeding it through ``reset``.
        super().reset(seed=seed)
        obs, info = self._env.reset()
        return np.clip(obs, -1.0, 1.0).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            np.clip(obs, -1.0, 1.0).astype(np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        return None

    def close(self):
        return None


def make_sac_env(
    image_path: str | Path | None,
    seed: int,
    success_coverage: float = 0.99,
):
    """Return a picklable SB3 environment factory."""

    def _init():
        return BeadTraceEnv(
            image_path=image_path,
            seed=seed,
            success_coverage=success_coverage,
        )

    return _init


class PaperTraceEnv(gym.Env):
    """25-D historical PPO environment used by the archived PPO checkpoints."""

    metadata = {"render_modes": []}

    def __init__(self, image_path: str | Path | None = None, seed: int = 42) -> None:
        super().__init__()
        self._env = PaperInspiredBeadEnvironment(image_path=image_path, seed=seed)
        self._image_path = image_path
        self._seed = seed
        self.observation_space = spaces.Box(
            -1.0,
            1.0,
            shape=(PaperInspiredBeadEnvironment.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        # Preserve the archived wrapper semantics: Gymnasium receives ``seed``
        # but the core environment continues from the RNG seeded at construction.
        # Historical training/evaluation code seeded each core env explicitly in
        # its constructor rather than reseeding it through ``reset``.
        super().reset(seed=seed)
        obs, info = self._env.reset()
        return np.clip(obs, -1.0, 1.0).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            np.clip(obs, -1.0, 1.0).astype(np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        return None

    def close(self):
        return None


def make_paper_env(image_path: str | Path | None, seed: int):
    def _init():
        return PaperTraceEnv(image_path=image_path, seed=seed)

    return _init


class MultiShapePaperTraceEnv(PaperTraceEnv):
    """PPO environment that samples a new contour image on every episode."""

    def __init__(self, image_paths: list[str], seed: int = 42) -> None:
        if not image_paths:
            raise ValueError("image_paths must be non-empty")
        self._image_paths = list(image_paths)
        self._rng = random.Random(seed)
        first = self._rng.choice(self._image_paths)
        super().__init__(image_path=first, seed=seed)

    def reset(self, *, seed: int | None = None, options=None):
        # Historical MultiShapeEnv used the constructor-seeded Python RNG and
        # intentionally did not reseed that RNG from Gymnasium's reset seed.
        image = self._rng.choice(self._image_paths)
        core_seed = self._rng.randint(0, 2**31 - 1)
        self._env = PaperInspiredBeadEnvironment(image_path=image, seed=core_seed)
        self._image_path = image
        self._seed = core_seed
        gym.Env.reset(self, seed=seed)
        obs, info = self._env.reset()
        return np.clip(obs, -1.0, 1.0).astype(np.float32), info


def make_multishape_paper_env(image_paths: list[str], seed: int):
    def _init():
        return MultiShapePaperTraceEnv(image_paths=image_paths, seed=seed)

    return _init
