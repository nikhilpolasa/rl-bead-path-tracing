#!/usr/bin/env python
"""Render one deterministic rollout with dual coverage metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC

from bead_rl.gym_envs import BeadTraceEnv, PaperTraceEnv
from bead_rl.metrics import points_to_polyline_distance
from bead_rl.policies import EluMlpExtractor  # noqa: F401 - needed for historical PPO load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["sac", "ppo"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--out", default="results/generated/trajectory.png")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.algorithm == "sac":
        model = SAC.load(args.model, device=args.device)
        env = BeadTraceEnv(args.image, seed=args.seed, success_coverage=0.99)
    else:
        model = PPO.load(args.model, device=args.device)
        env = PaperTraceEnv(args.image, seed=args.seed)

    obs, _ = env.reset()
    core = env._env
    path = np.asarray(core.path_points, dtype=np.float64)
    breaks = set(core.path_breaks)
    trajectory = [(float(core.pos_x), float(core.pos_y))]
    total_reward = 0.0
    terminated = truncated = False
    info = {}
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        trajectory.append((float(core.pos_x), float(core.pos_y)))
    env.close()

    trajectory = np.asarray(trajectory, dtype=np.float64)
    d_reference = points_to_polyline_distance(path, trajectory)
    d_trajectory = points_to_polyline_distance(trajectory, path, path_breaks=breaks)
    geom10 = float(np.mean(d_reference <= 0.10))
    rmse = float(np.sqrt(np.mean(d_trajectory**2)))
    sequential = float(info.get("coverage", 0.0))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    start = 0
    for end in sorted(breaks) + [len(path)]:
        segment = path[start:end]
        if len(segment) > 1:
            ax.plot(segment[:, 0], segment[:, 1], "--", linewidth=1.5, label="Reference" if start == 0 else None)
        start = end
    ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2.0, label="Trajectory")
    ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], marker="o", s=55, label="Start")
    ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], marker="x", s=65, label="Finish")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Arena X")
    ax.set_ylabel("Arena Y")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title(
        f"Sequential={sequential:.2%} | Geom@0.10={geom10:.2%} | "
        f"RMSE={rmse:.4f} | Reward={total_reward:.1f}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"saved={output} sequential={sequential:.6f} geom10={geom10:.6f} "
        f"rmse={rmse:.6f} reward={total_reward:.6f} steps={len(trajectory)-1}"
    )


if __name__ == "__main__":
    main()
