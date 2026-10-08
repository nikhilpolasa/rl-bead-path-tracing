"""Historical paper-inspired PPO environment.

The PPO experiments adapted observation/reward ideas from a published 4WIS
mobile-robot controller to the BTP's point-mass contour-tracing task.  This is
*not* the same MDP as the SAC environment, so PPO-vs-SAC results must be read as
comparisons of training/controller pipelines rather than an algorithm-only
ablation.

Important reproducibility note
------------------------------
The archived PPO checkpoints were trained with a smoothing-term implementation
that, because the base ``step`` updates ``prev_action`` before the PPO reward is
computed, behaves as a first-order action-change penalty.  The early manuscript
draft described it as a second-order action difference.  This module preserves
what the code actually executed so archived checkpoints remain evaluable.  See
``docs/reproducibility_audit.md``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import BeadEnvironment


_PAPER_LOOKAHEAD_METRES = [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 12.5, 15.0]
_LOOKAHEAD_SCALE = 8.0 / 15.0
LOOKAHEAD_DISTANCES = [d * _LOOKAHEAD_SCALE for d in _PAPER_LOOKAHEAD_METRES]

W_PROG = 1.0
W_ALIGN = 0.8
W_LAT = 0.04
W_HEAD = 0.01
W_SMOOTH = 0.01
SIGMA_Y = 0.20
SIGMA_PSI = 0.35


class PaperInspiredBeadEnvironment(BeadEnvironment):
    """25-D historical PPO environment used to create the archived models."""

    OBS_DIM = 25

    def __init__(self, image_path: str | Path | None = None, seed: int = 42) -> None:
        super().__init__(image_path=image_path, seed=seed, success_coverage=0.99)
        self._arc_length_prev = 0.0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        _, info = super().reset(seed=seed)
        self._arc_length_prev = 0.0
        return self._paper_observe(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        # This is the previous action before the base step.  The historical code
        # named it prev_action2, although it is a_{t-1}, not a_{t-2}.
        previous_action_before_step = self.prev_action.copy()

        # Advance dynamics/coverage with the base environment and discard its reward.
        _, _, terminated, truncated, info = super().step(action)
        action_arr = self._coerce_action(action)

        reward = self._historical_paper_reward(
            action_arr,
            previous_action_before_step,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self._arc_length_prev = self._arc_length()
        return self._paper_observe(), float(reward), terminated, truncated, info

    def _paper_observe(self) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        if len(self.path_points) == 0:
            return obs

        waypoint = min(self.waypoint_idx, len(self.path_points) - 1)
        ahead_points = self.path_points[waypoint:]
        if len(ahead_points) > 1:
            segments = np.linalg.norm(np.diff(ahead_points, axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(segments)])
        else:
            cumulative = np.array([0.0])

        for i, distance in enumerate(LOOKAHEAD_DISTANCES):
            idx = int(np.clip(np.searchsorted(cumulative, distance), 0, len(ahead_points) - 1))
            target = ahead_points[idx]
            obs[2 * i] = float(np.clip((target[0] - self.pos_x) / self.ARENA, -1.0, 1.0))
            obs[2 * i + 1] = float(
                np.clip((target[1] - self.pos_y) / self.ARENA, -1.0, 1.0)
            )

        heading_error = self._heading_error()
        obs[18] = float(np.cos(heading_error))
        obs[19] = float(np.sin(heading_error))
        obs[20] = float(np.clip(self.vel_x / self.MAX_SPEED, -1.0, 1.0))
        obs[21] = float(np.clip(self.vel_y / self.MAX_SPEED, -1.0, 1.0))
        speed = float(np.hypot(self.vel_x, self.vel_y))
        obs[22] = float(np.clip(speed / self.MAX_SPEED, 0.0, 1.0))
        obs[23] = float(self.prev_action[0])
        obs[24] = float(self.prev_action[1])
        return obs

    def _historical_paper_reward(
        self,
        action: np.ndarray,
        previous_action_before_step: np.ndarray,
        *,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> float:
        if len(self.path_points) == 0:
            return 0.0

        arc_length = self._arc_length()
        delta_s = max(0.0, arc_length - self._arc_length_prev)
        r_progress = W_PROG * delta_s

        lateral_error = self._lateral_error()
        heading_error = self._heading_error()
        tangential_speed = self._tangential_speed()
        heading_gate = max(0.0, float(np.cos(heading_error)))
        lateral_decay = float(np.exp(-((lateral_error / SIGMA_Y) ** 2)))
        r_alignment = W_ALIGN * tangential_speed * self.DT * heading_gate * lateral_decay

        r_lateral = -W_LAT * ((lateral_error / SIGMA_Y) ** 2)
        r_heading = -W_HEAD * ((heading_error / SIGMA_PSI) ** 2)

        # Historical behavior: super().step() has already set self.prev_action=a_t.
        # Therefore the old expression `a_t - 2*prev_action + old_prev_action`
        # reduces to `old_prev_action - a_t`, a first-order action-change term.
        action_change = previous_action_before_step - action
        r_smooth = -W_SMOOTH * float(np.linalg.norm(action_change)) ** 2

        bonus = 0.0
        if terminated:
            bonus = self.R_COMPLETION
        elif truncated:
            coverage = float(info.get("coverage", 0.0))
            bonus = self.R_PARTIAL80 if coverage >= 0.80 else self.R_INCOMPLETE

        return float(
            r_progress
            + r_alignment
            + r_lateral
            + r_heading
            + r_smooth
            + bonus
        )

    def _arc_length(self) -> float:
        if len(self.path_points) < 2:
            return 0.0
        segment_lengths = np.linalg.norm(np.diff(self.path_points, axis=0), axis=1)
        mean_segment = float(np.mean(segment_lengths)) if len(segment_lengths) else 0.0
        return self.waypoint_idx * mean_segment

    def _lateral_error(self) -> float:
        _, distance = self._local_path_info()
        return float(distance)

    def _heading_error(self) -> float:
        nearest_idx, _ = self._local_path_info()
        if nearest_idx < 0 or len(self.path_points) < 2:
            return 0.0

        next_idx = min(nearest_idx + 1, len(self.path_points) - 1)
        prev_idx = max(nearest_idx - 1, 0)
        if next_idx == nearest_idx:
            tangent = self.path_points[nearest_idx] - self.path_points[prev_idx]
        elif prev_idx == nearest_idx:
            tangent = self.path_points[next_idx] - self.path_points[nearest_idx]
        else:
            tangent = self.path_points[next_idx] - self.path_points[prev_idx]
        tangent_angle = float(np.arctan2(tangent[1], tangent[0]))

        speed = float(np.hypot(self.vel_x, self.vel_y))
        if speed < 1e-6:
            waypoint = min(self.waypoint_idx + 1, len(self.path_points) - 1)
            dx = self.path_points[waypoint, 0] - self.pos_x
            dy = self.path_points[waypoint, 1] - self.pos_y
            heading_angle = float(np.arctan2(dy, dx))
        else:
            heading_angle = float(np.arctan2(self.vel_y, self.vel_x))

        error = heading_angle - tangent_angle
        return float((error + np.pi) % (2 * np.pi) - np.pi)

    def _tangential_speed(self) -> float:
        nearest_idx, _ = self._local_path_info()
        if nearest_idx < 0 or len(self.path_points) < 2:
            return 0.0

        next_idx = min(nearest_idx + 1, len(self.path_points) - 1)
        prev_idx = max(nearest_idx - 1, 0)
        if next_idx == nearest_idx:
            tangent = self.path_points[nearest_idx] - self.path_points[prev_idx]
        elif prev_idx == nearest_idx:
            tangent = self.path_points[next_idx] - self.path_points[nearest_idx]
        else:
            tangent = self.path_points[next_idx] - self.path_points[prev_idx]

        norm = float(np.linalg.norm(tangent))
        if norm < 1e-8:
            return 0.0
        tangent = tangent / norm
        return float(max(0.0, self.vel_x * tangent[0] + self.vel_y * tangent[1]))
