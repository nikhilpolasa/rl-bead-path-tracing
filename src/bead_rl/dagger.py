"""DAgger utilities used to refine the SAC actor."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .gym_envs import BeadTraceEnv


def expert_action(env: BeadTraceEnv) -> np.ndarray:
    """Hand-coded expert: unit acceleration toward the next reference waypoint."""
    core = env._env
    waypoint = min(core.waypoint_idx, len(core.path_points) - 1)
    target_idx = min(waypoint + 1, len(core.path_points) - 1)
    position = np.array([core.pos_x, core.pos_y], dtype=np.float32)
    vector = core.path_points[target_idx].astype(np.float32) - position
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return np.clip(vector / norm, -1.0, 1.0).astype(np.float32)


def actor_action(model, observation: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=model.device
        ).unsqueeze(0)
        action = model.actor(tensor, deterministic=True)[0]
    return action.detach().cpu().numpy().astype(np.float32)


def fit_actor(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int = 512,
) -> float:
    """Supervised actor-only regression used by the DAgger repair stage."""
    if len(observations) == 0:
        raise ValueError("Cannot fit actor on an empty dataset")

    actor = model.actor
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate, weight_decay=1e-6)
    x = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    y = torch.as_tensor(actions, dtype=torch.float32, device=model.device)

    final_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(len(x), device=model.device)
        total_loss = 0.0
        for start in range(0, len(x), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = actor(x[indices], deterministic=True)
            loss = F.mse_loss(prediction, y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
        final_loss = total_loss / len(x)
    actor.eval()
    return final_loss


def collect_expert(
    image: str,
    *,
    episodes: int,
    seed: int,
    success_coverage: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    env = BeadTraceEnv(
        image_path=image,
        seed=seed,
        success_coverage=success_coverage,
    )
    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not terminated and not truncated:
            action = expert_action(env)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, _, terminated, truncated, _ = env.step(action)
    env.close()
    return np.asarray(observations, np.float32), np.asarray(actions, np.float32)


def collect_dagger(
    model,
    image: str,
    *,
    episodes: int,
    beta: float,
    seed: int,
    success_coverage: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Collect learner-state data labelled by the expert.

    The behaviour policy follows the expert with probability ``beta`` and the
    current actor otherwise; every visited state is labelled with the expert
    action, which is the key DAgger mechanism.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    observations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    stats: list[dict[str, float]] = []
    env = BeadTraceEnv(
        image_path=image,
        seed=seed,
        success_coverage=success_coverage,
    )

    # Share the core RNG with the expert/policy mixing coin flip.  The archived
    # implementation used NumPy's global RandomState for both reset offsets and
    # beta sampling; using the core's local RandomState preserves that sequence
    # without leaking randomness across environments.
    rng = env._env.rng

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        info: dict = {}
        while not terminated and not truncated:
            expert = expert_action(env)
            learner = actor_action(model, obs)
            observations.append(obs.copy())
            labels.append(expert.copy())
            action = expert if float(rng.random_sample()) < beta else learner
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
        stats.append(
            {
                "sequential": float(info.get("coverage", 0.0)),
                "reward": total_reward,
            }
        )
    env.close()
    return np.asarray(observations, np.float32), np.asarray(labels, np.float32), stats
