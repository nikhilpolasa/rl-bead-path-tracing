#!/usr/bin/env python
"""Train the multi-shape PPO anti-overfitting curriculum."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from bead_rl.evaluation import run_episode
from bead_rl.gym_envs import MultiShapePaperTraceEnv, PaperTraceEnv, make_multishape_paper_env
from bead_rl.policies import ppo_policy_kwargs


def collect_images(directory: str) -> list[str]:
    root = Path(directory)
    images = [
        *root.glob("*.jpg"),
        *root.glob("*.jpeg"),
        *root.glob("*.png"),
        *root.glob("*.JPG"),
        *root.glob("*.JPEG"),
        *root.glob("*.PNG"),
    ]
    return [str(path) for path in sorted(set(images))]


def evaluate_images(model, images: list[str], episodes_per_image: int, seed_base: int):
    rows = []
    for image_index, image in enumerate(images):
        for episode_index in range(episodes_per_image):
            seed = seed_base + image_index * 100 + episode_index
            rows.append(run_episode(model, lambda i=image, s=seed: PaperTraceEnv(i, seed=s)))
    mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    std = {key: float(np.std([row[key] for row in rows])) for key in rows[0]}
    return mean, std


class ValidationCallback(BaseCallback):
    def __init__(self, images: list[str], eval_freq: int, best_path: str):
        super().__init__(verbose=1)
        self.images = images
        self.eval_freq = eval_freq
        self.best_path = best_path
        self.last_eval = 0
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        mean, _ = evaluate_images(
            self.model,
            self.images,
            episodes_per_image=2,
            seed_base=8000 + self.num_timesteps,
        )
        score = 0.60 * mean["geom10"] + 0.25 * mean["sequential"] - 0.15 * mean["rmse"]
        print(
            f"[multi-PPO @{self.num_timesteps:,}] seq={mean['sequential']:.2%} "
            f"geom@.10={mean['geom10']:.2%} rmse={mean['rmse']:.4f} score={score:.4f}"
        )
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.best_path)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/closed_shapes_dataset/train")
    parser.add_argument("--val-dir", default="data/closed_shapes_dataset/val")
    parser.add_argument("--out", default="models/ppo_multishape_final")
    parser.add_argument("--best", default="models/ppo_multishape_best")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=3_000_000,
        help="Steps for this invocation; when --resume is used these are additional steps.",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    np.random.seed(2026)
    torch.manual_seed(2026)
    train_images = collect_images(args.train_dir)
    val_images = collect_images(args.val_dir)
    if not train_images or not val_images:
        raise FileNotFoundError("Training and validation directories must contain images")
    if len(val_images) > 10:
        val_images = random.Random(42).sample(val_images, 10)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("results/checkpoints/ppo_multishape").mkdir(parents=True, exist_ok=True)

    env_fns = [make_multishape_paper_env(train_images, seed=i) for i in range(args.n_envs)]
    train_env = VecNormalize(
        VecMonitor(SubprocVecEnv(env_fns)),
        norm_obs=False,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )

    if args.resume:
        model = PPO.load(
            args.resume,
            env=train_env,
            device=args.device,
            learning_rate=args.lr,
        )
        reset_num_timesteps = False
    else:
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
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=ppo_policy_kwargs(),
            verbose=1,
            device=args.device,
        )
        reset_num_timesteps = True

    callbacks = [
        ValidationCallback(val_images, args.eval_every, args.best),
        CheckpointCallback(
            save_freq=200_000,
            save_path="results/checkpoints/ppo_multishape",
            name_prefix="ppo_multishape",
        ),
    ]
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(args.out)
    train_env.save(f"{args.out}_vecnorm.pkl")
    train_env.close()

    best_path = f"{args.best}.zip"
    best = PPO.load(best_path if Path(best_path).exists() else f"{args.out}.zip", device=args.device)
    mean, std = evaluate_images(best, val_images[:5], episodes_per_image=4, seed_base=9000)
    summary = {"mean": mean, "std": std, "episodes_per_image": 4, "images": val_images[:5]}
    Path("results/ppo_multishape_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
