"""Procedural closed-shape dataset generation utilities."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle, RegularPolygon


SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def generate_closed_shape(path: str | Path, rng: np.random.RandomState) -> dict[str, object]:
    """Render one random closed contour and return its generation metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(4, 4), dpi=100)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    shape_type = str(rng.choice(["rectangle", "ellipse", "polygon"]))
    line_width = float(rng.uniform(2.0, 5.0))
    center_x = float(rng.uniform(0.4, 0.6))
    center_y = float(rng.uniform(0.4, 0.6))

    metadata: dict[str, object] = {
        "shape_type": shape_type,
        "line_width": line_width,
        "center_x": center_x,
        "center_y": center_y,
    }

    if shape_type == "rectangle":
        width = float(rng.uniform(0.3, 0.7))
        height = width if float(rng.rand()) < 0.3 else float(rng.uniform(0.3, 0.7))
        angle = float(rng.uniform(0, 360))
        patch = Rectangle(
            (center_x - width / 2, center_y - height / 2),
            width,
            height,
            angle=angle,
            rotation_point="center",
            edgecolor="black",
            facecolor="none",
            linewidth=line_width,
        )
        metadata.update(width=width, height=height, angle=angle)
    elif shape_type == "ellipse":
        width = float(rng.uniform(0.3, 0.7))
        height = width if float(rng.rand()) < 0.3 else float(rng.uniform(0.3, 0.7))
        angle = float(rng.uniform(0, 360))
        patch = Ellipse(
            (center_x, center_y),
            width,
            height,
            angle=angle,
            edgecolor="black",
            facecolor="none",
            linewidth=line_width,
        )
        metadata.update(width=width, height=height, angle=angle)
    else:
        vertices = int(rng.randint(3, 7))
        radius = float(rng.uniform(0.2, 0.4))
        orientation = float(rng.uniform(0, 2 * np.pi))
        patch = RegularPolygon(
            (center_x, center_y),
            numVertices=vertices,
            radius=radius,
            orientation=orientation,
            edgecolor="black",
            facecolor="none",
            linewidth=line_width,
        )
        metadata.update(vertices=vertices, radius=radius, orientation=orientation)

    axis.add_patch(patch)
    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return metadata


def generate_dataset(
    output_dir: str | Path,
    *,
    total_images: int = 1000,
    seed: int = 2026,
) -> Path:
    """Generate a 70/15/15 closed-shape train/validation/test dataset.

    The original BTP dataset was generated without a recorded RNG seed, so the
    exact historical images are shipped separately in ``data/closed_shapes_dataset``.
    This function provides a deterministic generator for future experiments.
    """
    if total_images <= 0:
        raise ValueError("total_images must be positive")

    output = Path(output_dir)
    rng = np.random.RandomState(seed)
    train_count = int(total_images * SPLITS["train"])
    val_count = int(total_images * SPLITS["val"])
    counts = {
        "train": train_count,
        "val": val_count,
        "test": total_images - train_count - val_count,
    }

    rows: list[dict[str, object]] = []
    for split, count in counts.items():
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            filename = f"shape_{split}_{index:04d}.jpg"
            metadata = generate_closed_shape(split_dir / filename, rng)
            rows.append({"split": split, "filename": filename, **metadata})

    manifest = output / "manifest.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest
