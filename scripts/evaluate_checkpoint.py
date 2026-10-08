#!/usr/bin/env python
"""Evaluate one archived or newly trained checkpoint with dual coverage metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import SAC

from bead_rl.evaluation import evaluate
from bead_rl.gym_envs import BeadTraceEnv, PaperTraceEnv
from bead_rl.policies import load_archived_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["sac", "ppo"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--success-coverage", type=float, default=0.99)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.algorithm == "sac":
        model = SAC.load(args.model, device=args.device)

        def env_for_seed(seed: int):
            return BeadTraceEnv(
                args.image,
                seed=seed,
                success_coverage=args.success_coverage,
            )

    else:
        model = load_archived_ppo(args.model, device=args.device)

        def env_for_seed(seed: int):
            return PaperTraceEnv(args.image, seed=seed)

    mean, std, rows = evaluate(
        model,
        env_for_seed,
        episodes=args.episodes,
        seed_base=args.seed,
    )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "algorithm": args.algorithm,
        "model": args.model,
        "image": args.image,
        "episodes": args.episodes,
        "seed_base": args.seed,
        "mean": mean,
        "std": std,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
