"""Geometry-based evaluation metrics for contour-tracing rollouts."""
from __future__ import annotations

import numpy as np


def valid_segments(
    polyline: np.ndarray,
    path_breaks: set[int] | list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return segment start/end arrays while respecting disconnected path breaks.

    ``path_breaks`` stores the index of the first point of a new component.  A
    break at index ``k`` therefore removes the segment ``k-1 -> k``.
    """
    points = np.asarray(polyline, dtype=np.float64)
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    starts, ends = points[:-1], points[1:]
    if path_breaks:
        keep = np.ones(len(starts), dtype=bool)
        for value in path_breaks:
            index = int(value)
            if 1 <= index < len(points):
                keep[index - 1] = False
        starts, ends = starts[keep], ends[keep]
    return starts, ends


def points_to_polyline_distance(
    points: np.ndarray,
    polyline: np.ndarray,
    path_breaks: set[int] | list[int] | tuple[int, ...] | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute the minimum Euclidean distance from each point to a polyline.

    Distances are calculated by exact orthogonal projection onto every valid
    line segment, not only to discrete waypoints.  The batched implementation
    keeps memory bounded for long trajectories.
    """
    query = np.asarray(points, dtype=np.float64)
    line = np.asarray(polyline, dtype=np.float64)
    starts, ends = valid_segments(line, path_breaks)

    if len(starts) == 0:
        if len(line) == 0:
            return np.full(len(query), np.inf, dtype=np.float64)
        return np.linalg.norm(query - line[0], axis=1)

    segments = ends - starts
    denominator = np.maximum(np.sum(segments * segments, axis=1), 1e-12)
    result = np.empty(len(query), dtype=np.float64)

    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        relative = batch[:, None, :] - starts[None, :, :]
        alpha = np.sum(relative * segments[None, :, :], axis=2) / denominator[None, :]
        alpha = np.clip(alpha, 0.0, 1.0)
        projected = starts[None, :, :] + alpha[:, :, None] * segments[None, :, :]
        squared = np.sum((batch[:, None, :] - projected) ** 2, axis=2)
        result[start : start + len(batch)] = np.sqrt(np.min(squared, axis=1))
    return result


def rollout_metrics(
    reference_path: np.ndarray,
    trajectory: np.ndarray,
    *,
    sequential_coverage: float,
    path_breaks: set[int] | None = None,
    reward: float | None = None,
) -> dict[str, float]:
    """Compute the dual coverage protocol used by the BTP.

    * Geometric coverage measures reference-point coverage by the *entire*
      rollout trajectory.
    * Tracking error measures each rollout sample's distance to the reference
      polyline.

    Episode reward is accepted only as metadata because reward functions differ
    between the SAC and paper-inspired PPO pipelines and should not be used as a
    cross-pipeline accuracy metric.
    """
    reference = np.asarray(reference_path, dtype=np.float64)
    rollout = np.asarray(trajectory, dtype=np.float64)
    d_reference = points_to_polyline_distance(reference, rollout)
    d_trajectory = points_to_polyline_distance(
        rollout, reference, path_breaks=path_breaks
    )

    metrics = {
        "sequential": float(sequential_coverage),
        "geom05": float(np.mean(d_reference <= 0.05)),
        "geom10": float(np.mean(d_reference <= 0.10)),
        "geom15": float(np.mean(d_reference <= 0.15)),
        "rmse": float(np.sqrt(np.mean(d_trajectory**2))),
        "mean_error": float(np.mean(d_trajectory)),
        "p95_error": float(np.percentile(d_trajectory, 95)),
        "max_error": float(np.max(d_trajectory)),
        "reference_p95": float(np.percentile(d_reference, 95)),
        "reference_max": float(np.max(d_reference)),
        "steps": float(max(0, len(rollout) - 1)),
    }
    if reward is not None:
        metrics["reward"] = float(reward)
    return metrics
