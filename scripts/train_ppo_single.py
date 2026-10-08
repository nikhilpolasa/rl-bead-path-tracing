#!/usr/bin/env python
"""Train the historical single-contour PPO baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from bead_rl.evaluation import evaluate
from bead_rl.gym_envs import PaperTraceEnv, make_paper_env
from bead_rl.policies import ppo_policy_kwargs


class EvaluationCallback(BaseCallback):
    def __init__(self, image: str, eval_freq: int, best_path: str):
        super().__init__(verbose=1)
        self.image = image
        self.eval_freq = eval_freq
        self.best_path = best_path
        self.last_eval = 0
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps

        def env_for_seed(seed: int):
            return PaperTraceEnv(self.image, seed=seed)

        mean, _, _ = evaluate(
            self.model, env_for_seed, episodes=5, seed_base=8000 + self.num_timesteps
        )
        score = 0.70 * mean["geom10"] + 0.20 * mean["sequential"] - 0.10 * mean["rmse"]
        print(
            f"[PPO @{self.num_timesteps:,}] seq={mean['sequential']:.2%} "
            f"geom@.10={mean['geom10']:.2%} rmse={mean['rmse']:.4f}"
        )
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.best_path)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="models/ppo_single_final")
    parser.add_argument("--best", default="models/ppo_single_best")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    np.random.seed(2026)
    torch.manual_seed(2026)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("results/checkpoints/ppo_single").mkdir(parents=True, exist_ok=True)

    env_fns = [make_paper_env(args.image, seed=i) for i in range(args.n_envs)]
    train_env = VecNormalize(
        VecMonitor(SubprocVecEnv(env_fns)),
        norm_obs=False,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        gamma=0.99,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.01,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=ppo_policy_kwargs(),
        verbose=1,
        device=args.device,
    )
    callbacks = [
        EvaluationCallback(args.image, args.eval_every, args.best),
        CheckpointCallback(
            save_freq=100_000,
            save_path="results/checkpoints/ppo_single",
            name_prefix="ppo_single",
        ),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(args.out)
    train_env.save(f"{args.out}_vecnorm.pkl")
    train_env.close()

    best_path = f"{args.best}.zip"
    best = PPO.load(best_path if Path(best_path).exists() else f"{args.out}.zip", device=args.device)

    def env_for_seed(seed: int):
        return PaperTraceEnv(args.image, seed=seed)

    mean, std, _ = evaluate(best, env_for_seed, episodes=20, seed_base=9000)
    summary = {"mean": mean, "std": std, "episodes": 20}
    Path("results/ppo_single_training_image.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
