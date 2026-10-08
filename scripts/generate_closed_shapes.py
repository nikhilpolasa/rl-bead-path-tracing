#!/usr/bin/env python
from __future__ import annotations

import argparse

from bead_rl.datasets import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/generated_closed_shapes")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = generate_dataset(args.output, total_images=args.count, seed=args.seed)
    print(f"Generated dataset; manifest: {manifest}")


if __name__ == "__main__":
    main()
