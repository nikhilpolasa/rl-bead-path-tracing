#!/usr/bin/env python
"""Inspect SB3 checkpoint metadata without importing Stable-Baselines3."""
from __future__ import annotations

import argparse
import json
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    with zipfile.ZipFile(args.checkpoint) as archive:
        data = json.loads(archive.read("data"))
        version = archive.read("_stable_baselines3_version").decode().strip()
        selected = {
            "stable_baselines3": version,
            "num_timesteps": data.get("num_timesteps"),
            "observation_shape": data.get("observation_space", {}).get("_shape"),
            "action_shape": data.get("action_space", {}).get("_shape"),
            "learning_rate": data.get("learning_rate"),
            "gamma": data.get("gamma"),
            "buffer_size": data.get("buffer_size"),
            "batch_size": data.get("batch_size"),
            "n_steps": data.get("n_steps"),
            "n_epochs": data.get("n_epochs"),
            "ent_coef": data.get("ent_coef"),
            "vf_coef": data.get("vf_coef"),
            "seed": data.get("seed"),
        }
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
