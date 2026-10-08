"""RL contour-tracing research code for the B.Tech project."""

from .core import BeadEnvironment
from .metrics import points_to_polyline_distance, rollout_metrics, valid_segments

__all__ = [
    "BeadEnvironment",
    "points_to_polyline_distance",
    "rollout_metrics",
    "valid_segments",
]

__version__ = "0.1.0"
