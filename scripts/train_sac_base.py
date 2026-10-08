#!/usr/bin/env python
"""Reconstructed SAC base-training entry point.

The original from-scratch SAC script was missing from the archived working
folder.  Hyperparameters below were recovered directly from
``FINAL_BTP_MODEL_95pct.zip`` metadata (SB3 2.8.0): 30k steps, gamma=.995,
lr=5e-5, buffer=200k, batch=256, fixed entropy coefficient 5e-4, [128,128]
actor/critic networks, learning_starts=0, train_freq=1, gradient_steps=1.

The historical README states that the base model used 95% sequential coverage
for termination; this script therefore exposes that value explicitly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from bead_rl.gym_envs import BeadTraceEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="models/sac_base_95")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--success-coverage", type=float, default=0.95)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    env = BeadTraceEnv(
        image_path=args.image,
        seed=args.seed,
        success_coverage=args.success_coverage,
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=5e-5,
        buffer_size=200_000,
        learning_starts=0,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef=0.0005,
        target_update_interval=1,
        policy_kwargs={"net_arch": {"pi": [128, 128], "qf": [128, 128]}, "use_sde": False},
        seed=args.seed,
        verbose=1,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    env.close()
    print(f"Saved {args.out}.zip")


if __name__ == "__main__":
    main()
