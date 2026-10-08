#!/usr/bin/env python
"""Compare final SAC+DAgger, single-shape PPO and multi-shape PPO on one image.

Only geometry-based metrics are used for cross-pipeline ranking.  Episode reward
is reported in per-model evaluation files but is intentionally omitted here
because SAC and PPO use different reward functions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import SAC

from bead_rl.evaluation import evaluate
from bead_rl.gym_envs import BeadTraceEnv, PaperTraceEnv
from bead_rl.policies import load_archived_ppo


def evaluate_sac(path: str, image: str, episodes: int, seed: int, device: str):
    model = SAC.load(path, device=device)
    return evaluate(
        model,
        lambda s: BeadTraceEnv(image, seed=s, success_coverage=0.99),
        episodes=episodes,
        seed_base=seed,
    )[:2]


def evaluate_ppo(path: str, image: str, episodes: int, seed: int, device: str):
    model = load_archived_ppo(path, device=device)
    return evaluate(
        model,
        lambda s: PaperTraceEnv(image, seed=s),
        episodes=episodes,
        seed_base=seed,
    )[:2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/benchmarks/ood_circle.jpg")
    parser.add_argument("--sac", default="models/sac_dagger_final.zip")
    parser.add_argument("--ppo-single", default="models/ppo_single_final.zip")
    parser.add_argument("--ppo-multi", default="models/ppo_multishape_best.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/ood_circle_comparison_recomputed.json")
    args = parser.parse_args()

    results = {}
    for name, evaluator, checkpoint in [
        ("sac_dagger", evaluate_sac, args.sac),
        ("ppo_single", evaluate_ppo, args.ppo_single),
        ("ppo_multishape", evaluate_ppo, args.ppo_multi),
    ]:
        if not Path(checkpoint).exists():
            results[name] = {"error": f"missing checkpoint: {checkpoint}"}
            continue
        mean, std = evaluator(
            checkpoint, args.image, args.episodes, args.seed, args.device
        )
        results[name] = {"mean": mean, "std": std}

    payload = {
        "image": args.image,
        "episodes": args.episodes,
        "seed_base": args.seed,
        "note": "Compare sequential/geometric coverage and RMSE; rewards differ by pipeline and are not cross-comparable.",
        "models": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
