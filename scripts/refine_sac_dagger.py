#!/usr/bin/env python
"""Refine a pretrained SAC actor with DAgger on the final 99% environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from bead_rl.dagger import collect_dagger, collect_expert, fit_actor
from bead_rl.evaluation import evaluate, model_selection_score
from bead_rl.gym_envs import BeadTraceEnv


def target_met(stats: dict[str, float]) -> bool:
    return (
        stats["sequential"] >= 0.99
        and stats["geom10"] >= 0.95
        and stats["rmse"] <= 0.05
        and 100.0 <= stats.get("reward", 0.0) <= 200.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--base", default="models/sac_base_95.zip")
    parser.add_argument("--out", default="models/sac_dagger_final")
    parser.add_argument("--summary", default="results/sac_dagger_training_validation.json")
    parser.add_argument("--expert-episodes", type=int, default=8)
    parser.add_argument("--dagger-episodes", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not Path(args.base).exists():
        raise FileNotFoundError(args.base)

    np.random.seed(2026)
    torch.manual_seed(2026)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)

    env = BeadTraceEnv(args.image, seed=42, success_coverage=0.99)
    model = SAC.load(args.base, env=env, device=args.device)

    anchor_obs, anchor_actions = collect_expert(
        args.image, episodes=args.expert_episodes, seed=1000
    )
    aggregate_obs = anchor_obs.copy()
    aggregate_actions = anchor_actions.copy()
    loss = fit_actor(
        model,
        aggregate_obs,
        aggregate_actions,
        epochs=8,
        learning_rate=1e-4,
    )

    def env_for_seed(seed: int):
        return BeadTraceEnv(args.image, seed=seed, success_coverage=0.99)

    mean, _, _ = evaluate(model, env_for_seed, episodes=5, seed_base=3000)
    best_score = model_selection_score(mean)
    model.save(args.out)
    print(
        f"anchor loss={loss:.6f} seq={mean['sequential']:.2%} "
        f"geom@.10={mean['geom10']:.2%} rmse={mean['rmse']:.4f}"
    )

    betas = [0.50, 0.25, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00]
    consecutive_hits = 0
    for round_index in range(1, args.rounds + 1):
        beta = betas[min(round_index - 1, len(betas) - 1)]
        obs, actions, _ = collect_dagger(
            model,
            args.image,
            episodes=args.dagger_episodes,
            beta=beta,
            seed=4000 + round_index * 100,
        )
        aggregate_obs = np.concatenate([aggregate_obs, obs], axis=0)
        aggregate_actions = np.concatenate([aggregate_actions, actions], axis=0)

        learning_rate = 1e-4 if round_index <= 3 else 5e-5
        loss = fit_actor(
            model,
            aggregate_obs,
            aggregate_actions,
            epochs=6,
            learning_rate=learning_rate,
        )
        mean, _, _ = evaluate(
            model,
            env_for_seed,
            episodes=6,
            seed_base=5000 + round_index * 100,
        )
        score = model_selection_score(mean)
        print(
            f"round={round_index} beta={beta:.2f} samples={len(aggregate_obs):,} "
            f"seq={mean['sequential']:.2%} geom@.10={mean['geom10']:.2%} "
            f"rmse={mean['rmse']:.4f} loss={loss:.6f} score={score:.4f}"
        )
        if score > best_score:
            best_score = score
            model.save(args.out)
        consecutive_hits = consecutive_hits + 1 if target_met(mean) else 0
        if consecutive_hits >= 2:
            break

    best = SAC.load(f"{args.out}.zip", device=args.device)
    final_mean, final_std, _ = evaluate(best, env_for_seed, episodes=20, seed_base=9000)
    payload = {"mean": final_mean, "std": final_std, "episodes": 20}
    Path(args.summary).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    env.close()


if __name__ == "__main__":
    main()
