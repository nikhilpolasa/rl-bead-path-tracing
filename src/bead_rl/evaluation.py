"""Shared rollout and aggregation helpers."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .metrics import rollout_metrics


def run_episode(model, env_factory: Callable[[], object]) -> dict[str, float]:
    """Run one deterministic policy episode and compute geometry metrics."""
    env = env_factory()
    obs, _ = env.reset()
    core = env._env
    reference = np.asarray(core.path_points, dtype=np.float64)
    breaks = set(core.path_breaks)
    trajectory = [(float(core.pos_x), float(core.pos_y))]

    total_reward = 0.0
    terminated = truncated = False
    info: dict = {}
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        trajectory.append((float(core.pos_x), float(core.pos_y)))

    env.close()
    return rollout_metrics(
        reference,
        np.asarray(trajectory, dtype=np.float64),
        sequential_coverage=float(info.get("coverage", 0.0)),
        path_breaks=breaks,
        reward=total_reward,
    )


def evaluate(
    model,
    env_factory_for_seed: Callable[[int], object],
    *,
    episodes: int,
    seed_base: int,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]]]:
    """Evaluate a model over repeated seeded episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rows = [
        run_episode(model, lambda seed=seed_base + i: env_factory_for_seed(seed))
        for i in range(episodes)
    ]
    keys = rows[0].keys()
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    std = {key: float(np.std([row[key] for row in rows])) for key in keys}
    return mean, std, rows


def model_selection_score(metrics: dict[str, float]) -> float:
    """Accuracy-weighted score used by the SAC+DAgger repair pipeline."""
    return 0.70 * metrics["geom10"] + 0.20 * metrics["sequential"] - 0.10 * metrics["rmse"]
