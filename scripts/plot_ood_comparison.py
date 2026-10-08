#!/usr/bin/env python
"""Plot the verified OOD comparison without mixing incompatible rewards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/ood_circle_comparison_verified.json")
    parser.add_argument("--out", default="results/generated/ood_comparison.png")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    keys = ["sac_dagger", "ppo_multishape", "ppo_single"]
    labels = ["SAC+DAgger", "Multi-shape PPO", "Single-shape PPO"]
    sequential = [100.0 * payload[key]["mean"]["sequential"] for key in keys]
    geometric = [100.0 * payload[key]["mean"]["geom10"] for key in keys]
    rmse = [payload[key]["mean"]["rmse"] for key in keys]

    x = np.arange(len(labels))
    width = 0.34
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, sequential, width, label="Sequential coverage (%)")
    ax.bar(x + width / 2, geometric, width, label="Geometric coverage @0.10 (%)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Out-of-distribution contour tracing")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    rmse_output = output.with_name(output.stem + "_rmse" + output.suffix)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, rmse)
    ax.set_ylabel("Tracking RMSE (arena units)")
    ax.set_title("Out-of-distribution tracking error")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(rmse_output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved={output} and {rmse_output}")


if __name__ == "__main__":
    main()
