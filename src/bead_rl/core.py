"""Core 2-D bead dynamics and raster-to-contour processing.

This module contains the environment logic used by the SAC + DAgger pipeline.
The implementation is a cleaned, reproducible version of the BTP code while
preserving the historical dynamics/reward design used by the trained models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


class BeadEnvironment:
    """Continuous 2-D point-mass environment for image-derived path tracing.

    The class intentionally does not inherit from Gymnasium.  It can therefore
    be unit-tested without RL dependencies; :class:`bead_rl.gym_envs.BeadTraceEnv`
    exposes the Gymnasium API used by Stable-Baselines3.

    Parameters
    ----------
    image_path:
        Path to one raster image or to a directory of raster images.  If a
        directory is supplied, a shape is sampled at each reset.
    seed:
        Seed for this environment's *local* NumPy RNG.  Historical code used
        ``np.random.seed`` globally; using ``RandomState`` preserves the same
        random-number algorithm while avoiding cross-environment interference.
    success_coverage:
        Sequential coverage required for successful termination.  ``0.99`` is
        the final SAC+DAgger setting.  The historical SAC base checkpoint was
        trained with ``0.95``.
    """

    ARENA = 10.0
    BEAD_R = 0.15
    BOUNDARY = ARENA - BEAD_R
    MAX_STEPS = 2000

    # Physics.
    ACCEL = 10.0
    MAX_SPEED = 7.0
    FRICTION = 7.0
    BOUNCE = 0.15
    DT = 0.016

    # Coverage.
    COV_RADIUS = 0.18
    STRICT_RADIUS = 0.10

    # Reward weights.
    R_APPROACH = 0.08
    R_WAYPOINT = 0.35
    R_TRACK_ERR = -0.015
    R_OFF_CURVE = -0.03
    R_BACKTRACK = -0.02
    R_BOUNDARY = -0.25
    R_SMOOTH = -0.001
    R_STEP = -0.001
    R_COMPLETION = 25.0
    R_PARTIAL80 = 10.0
    R_INCOMPLETE = -2.0

    OBS_DIM = 16

    def __init__(
        self,
        image_path: str | Path | None = None,
        seed: int = 42,
        success_coverage: float = 0.99,
    ) -> None:
        if not 0.0 < success_coverage <= 1.0:
            raise ValueError("success_coverage must be in (0, 1]")

        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.success_coverage = float(success_coverage)

        self.pos_x = self.pos_y = 0.0
        self.vel_x = self.vel_y = 0.0
        self.step_count = 0

        self.path_points = np.empty((0, 2), dtype=np.float64)
        self.path_breaks: set[int] = set()
        self.covered_points: set[int] = set()
        self.strict_covered_points: set[int] = set()

        self.waypoint_idx = 0
        self.prev_dist = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.hit_boundary = False
        self.no_progress_steps = 0
        self.last_reward_components = self._empty_reward_components()

        self.image_paths: list[Path] = []
        if image_path is None:
            self._create_default_path()
        else:
            self.image_paths = self._resolve_images(image_path)
            if not self.image_paths:
                raise FileNotFoundError(f"No supported images found at: {image_path}")
            self.load_contours(self._sample_image())

    def reseed(self, seed: int) -> None:
        """Reset this environment's local RNG."""
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

    @staticmethod
    def _resolve_images(path: str | Path) -> list[Path]:
        p = Path(path)
        if p.is_file():
            return [p]
        if not p.is_dir():
            return []
        images: list[Path] = []
        for pattern in _IMAGE_GLOBS:
            images.extend(p.glob(pattern))
        # Sorting makes directory enumeration deterministic; sampling is still random.
        return sorted(set(images))

    def _sample_image(self) -> Path:
        idx = int(self.rng.randint(0, len(self.image_paths)))
        return self.image_paths[idx]

    def _create_default_path(self) -> None:
        t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        self.path_points = np.stack([4.0 * np.cos(t), 4.0 * np.sin(t)], axis=1)
        self.path_breaks = set()

    def load_contours(self, image_path: str | Path) -> None:
        """Extract one or more ordered paths from a raster image."""
        p = Path(image_path)
        if p.is_dir():
            candidates = self._resolve_images(p)
            if not candidates:
                raise FileNotFoundError(f"No supported images found in: {p}")
            p = candidates[int(self.rng.randint(0, len(candidates)))]

        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f"OpenCV could not read image: {p}")

        image = self._resize_img(image)
        height, width = image.shape[:2]
        foreground = self._foreground_mask(image)
        foreground_ratio = cv2.countNonZero(foreground) / max(1, foreground.size)

        # Thin strokes are skeletonised; filled shapes use their external contour.
        paths = self._extract_stroke_paths(foreground) if foreground_ratio <= 0.20 else []
        if not paths:
            paths = self._extract_contour_paths(foreground)
        if not paths:
            raise ValueError(f"No usable contour could be extracted from: {p}")

        candidates: list[tuple[float, np.ndarray]] = []
        for path in paths:
            arena_path = self._to_arena(path, width, height)
            length = self._path_length(arena_path)
            if length >= 0.1:
                candidates.append((length, arena_path))
        if not candidates:
            raise ValueError(f"Extracted contours were too short in: {p}")

        candidates.sort(key=lambda item: item[0], reverse=True)
        total_length = sum(length for length, _ in candidates)

        resampled: list[np.ndarray] = []
        breaks: set[int] = set()
        running = 0
        for length, points in candidates:
            n_points = max(20, int(350 * length / total_length))
            rp = self._resample(points, n_points)
            if resampled:
                breaks.add(running)
            resampled.append(rp)
            running += len(rp)

        self.path_points = np.vstack(resampled).astype(np.float64)
        self.path_breaks = breaks

    @staticmethod
    def _resize_img(image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= max_dim:
            return image
        scale = max_dim / longest
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _foreground_mask(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        border = np.concatenate(
            [blurred[0, :], blurred[-1, :], blurred[:, 0], blurred[:, -1]]
        )
        threshold_mode = (
            cv2.THRESH_BINARY_INV if np.median(border) >= 127 else cv2.THRESH_BINARY
        )
        _, foreground = cv2.threshold(
            blurred, 0, 255, threshold_mode + cv2.THRESH_OTSU
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        return cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    def _extract_stroke_paths(self, foreground: np.ndarray) -> list[np.ndarray]:
        skeleton = self._zhang_suen(foreground)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        paths: list[np.ndarray] = []
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < 10:
                continue
            mask = (labels == component).astype(np.uint8)
            ordered = self._order_skeleton(mask)
            if len(ordered) >= 2:
                paths.append(ordered)
        return paths

    @staticmethod
    def _extract_contour_paths(foreground: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        image_area = foreground.shape[0] * foreground.shape[1]
        paths: list[np.ndarray] = []
        for contour in contours:
            if cv2.arcLength(contour, True) < 10:
                continue
            if cv2.contourArea(contour) < image_area * 0.0005:
                continue
            points = contour.reshape(-1, 2).astype(np.float64)
            if not np.allclose(points[0], points[-1]):
                points = np.vstack([points, points[0]])
            paths.append(points)
        return paths

    @staticmethod
    def _zhang_suen(foreground: np.ndarray, max_iter: int = 200) -> np.ndarray:
        """Vectorised Zhang-Suen thinning used by the original project."""
        image = (foreground > 0).astype(np.uint8)
        image[[0, -1], :] = 0
        image[:, [0, -1]] = 0

        def transition(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return ((a == 0) & (b == 1)).astype(np.uint8)

        for _ in range(max_iter):
            changed = False
            for step in (0, 1):
                p2 = image[:-2, 1:-1]
                p3 = image[:-2, 2:]
                p4 = image[1:-1, 2:]
                p5 = image[2:, 2:]
                p6 = image[2:, 1:-1]
                p7 = image[2:, :-2]
                p8 = image[1:-1, :-2]
                p9 = image[:-2, :-2]
                p1 = image[1:-1, 1:-1]
                neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                transitions = (
                    transition(p2, p3)
                    + transition(p3, p4)
                    + transition(p4, p5)
                    + transition(p5, p6)
                    + transition(p6, p7)
                    + transition(p7, p8)
                    + transition(p8, p9)
                    + transition(p9, p2)
                )
                if step == 0:
                    remove = (
                        (p1 == 1)
                        & (neighbours >= 2)
                        & (neighbours <= 6)
                        & (transitions == 1)
                        & ((p2 * p4 * p6) == 0)
                        & ((p4 * p6 * p8) == 0)
                    )
                else:
                    remove = (
                        (p1 == 1)
                        & (neighbours >= 2)
                        & (neighbours <= 6)
                        & (transitions == 1)
                        & ((p2 * p4 * p8) == 0)
                        & ((p2 * p6 * p8) == 0)
                    )
                if np.any(remove):
                    image[1:-1, 1:-1][remove] = 0
                    changed = True
            if not changed:
                break
        return image

    @staticmethod
    def _order_skeleton(mask: np.ndarray) -> np.ndarray:
        pixels = [tuple(map(int, p)) for p in np.argwhere(mask > 0)]
        if len(pixels) <= 1:
            return np.empty((0, 2), dtype=np.float64)

        pixel_set = set(pixels)
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),             (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]

        def neighbours(point: tuple[int, int]) -> list[tuple[int, int]]:
            y, x = point
            return [
                q for q in ((y + dy, x + dx) for dy, dx in offsets) if q in pixel_set
            ]

        degrees = {point: len(neighbours(point)) for point in pixels}
        endpoints = [point for point, degree in degrees.items() if degree == 1]
        start = min(endpoints or pixels, key=lambda point: (point[1], point[0]))

        path = [start]
        visited = {start}
        previous: tuple[int, int] | None = None
        current = start

        for _ in range(len(pixels) + 2):
            candidates = [q for q in neighbours(current) if q != previous]
            unvisited = [q for q in candidates if q not in visited]
            if not unvisited:
                if not endpoints and start in candidates and len(path) > 2:
                    path.append(start)
                break

            def score(next_point: tuple[int, int]) -> tuple[float, float]:
                if previous is None:
                    return float(next_point[1]), float(next_point[0])
                dy0, dx0 = current[0] - previous[0], current[1] - previous[1]
                dy1, dx1 = next_point[0] - current[0], next_point[1] - current[1]
                norm0 = max(1e-6, float(np.hypot(dy0, dx0)))
                norm1 = max(1e-6, float(np.hypot(dy1, dx1)))
                cosine = (dy0 * dy1 + dx0 * dx1) / (norm0 * norm1)
                return -float(cosine), norm1

            nxt = min(unvisited, key=score)
            previous, current = current, nxt
            path.append(current)
            visited.add(current)

        return np.array([[x, y] for y, x in path], dtype=np.float64)

    def _to_arena(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        scale = (2 * self.BOUNDARY) / max(1, width - 1, height - 1)
        result = np.empty_like(points, dtype=np.float64)
        result[:, 0] = (points[:, 0] - (width - 1) / 2) * scale
        result[:, 1] = ((height - 1) / 2 - points[:, 1]) * scale
        return result

    @staticmethod
    def _path_length(points: np.ndarray) -> float:
        if len(points) <= 1:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    @staticmethod
    def _resample(points: np.ndarray, n_points: int) -> np.ndarray:
        if len(points) <= 1:
            return points
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = cumulative[-1]
        if total == 0:
            return points
        targets = np.linspace(0.0, total, n_points)
        indices = np.clip(np.searchsorted(cumulative, targets) - 1, 0, len(points) - 2)
        denom = cumulative[indices + 1] - cumulative[indices]
        alpha = np.where(
            denom > 0,
            (targets - cumulative[indices]) / (denom + 1e-12),
            0.0,
        )
        return points[indices] + alpha[:, None] * (points[indices + 1] - points[indices])

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.reseed(seed)
        if len(self.image_paths) > 1:
            self.load_contours(self._sample_image())

        if len(self.path_points) > 0:
            self.pos_x = float(self.path_points[0, 0]) + float(self.rng.uniform(-0.05, 0.05))
            self.pos_y = float(self.path_points[0, 1]) + float(self.rng.uniform(-0.05, 0.05))
        else:
            self.pos_x = self.pos_y = 0.0

        self.vel_x = self.vel_y = 0.0
        self.step_count = 0
        self.waypoint_idx = 0
        self.covered_points.clear()
        self.strict_covered_points.clear()
        if len(self.path_points) > 0:
            self.covered_points.add(0)
            self.strict_covered_points.add(0)
        self.no_progress_steps = 0
        self.hit_boundary = False
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_reward_components = self._empty_reward_components()

        observation = self._observe()
        self.prev_dist = self._dist_to_waypoint()
        return observation, {}

    def step(self, action: np.ndarray | int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_arr = self._coerce_action(action)
        self._physics(action_arr)
        self.step_count += 1

        curr_dist = self._dist_to_waypoint()
        old_waypoint = self.waypoint_idx

        best_waypoint = old_waypoint
        upper = min(len(self.path_points) - 1, old_waypoint + 4)
        for candidate in range(old_waypoint, upper + 1):
            if candidate in self.path_breaks:
                continue
            distance = float(
                np.hypot(
                    self.path_points[candidate, 0] - self.pos_x,
                    self.path_points[candidate, 1] - self.pos_y,
                )
            )
            if distance <= self.COV_RADIUS:
                best_waypoint = candidate

        self.waypoint_idx = max(old_waypoint, best_waypoint)
        advanced_count = self.waypoint_idx - old_waypoint
        if advanced_count > 0:
            self.covered_points.update(range(old_waypoint + 1, self.waypoint_idx + 1))
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        nearest_idx, nearest_dist = self._local_path_info()
        if nearest_dist <= self.STRICT_RADIUS:
            self.strict_covered_points.add(int(nearest_idx))

        reward, components = self._reward(
            action_arr,
            curr_dist,
            advanced_count,
            nearest_idx,
            nearest_dist,
        )

        coverage = self.get_coverage()
        strict_coverage = len(self.strict_covered_points) / max(1, len(self.path_points))
        terminated = bool(coverage >= self.success_coverage and self.step_count > 100)
        stalled = self.no_progress_steps >= 400
        truncated = bool((self.step_count >= self.MAX_STEPS or stalled) and not terminated)

        if terminated:
            reward += self.R_COMPLETION
            components["completion"] = self.R_COMPLETION
            components["total"] = reward
        elif truncated:
            terminal_reward = self.R_PARTIAL80 if coverage >= 0.80 else self.R_INCOMPLETE
            key = "partial80" if coverage >= 0.80 else "incomplete"
            components[key] = terminal_reward
            reward += terminal_reward
            components["total"] = reward

        self.prev_dist = self._dist_to_waypoint()
        self.prev_action = action_arr

        info = {
            "coverage": float(coverage),
            "strict_coverage": float(strict_coverage),
            "tracking_error": float(nearest_dist),
            "on_canvas": not self.hit_boundary,
            "distance": float(self.prev_dist),
            "waypoint_idx": int(self.waypoint_idx),
            "advanced_count": int(advanced_count),
            "reward_components": components,
            "success": bool(coverage >= 0.80),
            "stalled": bool(stalled),
            "no_progress_steps": int(self.no_progress_steps),
        }
        return self._observe(), float(reward), terminated, truncated, info

    @staticmethod
    def _coerce_action(action: np.ndarray | int) -> np.ndarray:
        if np.isscalar(action) or (hasattr(action, "__len__") and len(action) == 1):
            mapping = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0), 4: (0, 0)}
            return np.array(mapping.get(int(action), (0, 0)), dtype=np.float32)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    def _observe(self) -> np.ndarray:
        if len(self.path_points) == 0:
            return np.zeros(self.OBS_DIM, dtype=np.float32)

        norm = self.ARENA
        waypoint = min(self.waypoint_idx, len(self.path_points) - 1)
        current_delta = (self.path_points[waypoint] - np.array([self.pos_x, self.pos_y])) / norm

        next_waypoint = min(waypoint + 1, len(self.path_points) - 1)
        next_delta = (self.path_points[next_waypoint] - np.array([self.pos_x, self.pos_y])) / norm

        lookahead_idx = min(waypoint + 5, len(self.path_points) - 1)
        lookahead = self.path_points[lookahead_idx] - np.array([self.pos_x, self.pos_y])
        lookahead_norm = max(1e-6, float(np.linalg.norm(lookahead)))

        nearest_idx, nearest_dist = self._local_path_info()
        on_curve = 1.0 if nearest_dist < self.COV_RADIUS else 0.0

        if 1 <= nearest_idx <= len(self.path_points) - 2:
            v1 = self.path_points[nearest_idx] - self.path_points[nearest_idx - 1]
            v2 = self.path_points[nearest_idx + 1] - self.path_points[nearest_idx]
            curvature = float(v1[0] * v2[1] - v1[1] * v2[0]) / (
                float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
            )
        else:
            curvature = 0.0

        speed = float(np.hypot(self.vel_x, self.vel_y))
        return np.array(
            [
                current_delta[0],
                current_delta[1],
                next_delta[0],
                next_delta[1],
                self.pos_x / norm,
                self.pos_y / norm,
                self.vel_x / self.MAX_SPEED,
                self.vel_y / self.MAX_SPEED,
                float(np.clip(nearest_dist / norm, 0.0, 1.0)),
                on_curve,
                float(np.clip(self.vel_y / (speed + 1e-8), -1.0, 1.0)),
                float(np.clip(self.vel_x / (speed + 1e-8), -1.0, 1.0)),
                float(np.clip(curvature, -1.0, 1.0)),
                lookahead[0] / lookahead_norm,
                lookahead[1] / lookahead_norm,
                float(self.waypoint_idx / max(1, len(self.path_points) - 1)),
            ],
            dtype=np.float32,
        )

    def _dist_to_waypoint(self) -> float:
        if len(self.path_points) == 0:
            return 0.0
        waypoint = min(self.waypoint_idx, len(self.path_points) - 1)
        return float(
            np.hypot(
                self.path_points[waypoint, 0] - self.pos_x,
                self.path_points[waypoint, 1] - self.pos_y,
            )
        )

    def _local_path_info(self, window: int = 20) -> tuple[int, float]:
        if len(self.path_points) == 0:
            return -1, self.ARENA * 2.0
        low = max(0, self.waypoint_idx - window)
        high = min(len(self.path_points), self.waypoint_idx + window + 1)
        points = self.path_points[low:high]
        distances = np.hypot(points[:, 0] - self.pos_x, points[:, 1] - self.pos_y)
        offset = int(np.argmin(distances))
        return low + offset, float(distances[offset])

    def _reward(
        self,
        action: np.ndarray,
        curr_dist: float,
        advanced_count: int = 0,
        nearest_idx: int | None = None,
        nearest_dist: float | None = None,
    ) -> tuple[float, dict[str, float]]:
        components = self._empty_reward_components()
        if len(self.path_points) == 0:
            return 0.0, components

        delta_norm = (self.prev_dist - curr_dist) / max(self.COV_RADIUS, 1e-6)
        components["approach"] = self.R_APPROACH * float(np.clip(delta_norm, -1.0, 1.0))
        components["waypoint"] = self.R_WAYPOINT * float(max(0, advanced_count))

        if nearest_idx is None or nearest_dist is None:
            nearest_idx, nearest_dist = self._local_path_info()

        error_norm = float(
            np.clip(nearest_dist / max(self.STRICT_RADIUS, 1e-6), 0.0, 2.0)
        )
        components["tracking_error"] = self.R_TRACK_ERR * error_norm
        if nearest_dist > self.COV_RADIUS:
            components["off_curve"] = self.R_OFF_CURVE
        if nearest_idx < self.waypoint_idx - 5 and nearest_dist <= self.COV_RADIUS:
            components["backtrack"] = self.R_BACKTRACK
        if self.hit_boundary:
            components["boundary"] = self.R_BOUNDARY

        action_delta = float(np.linalg.norm(action - self.prev_action))
        components["smooth"] = self.R_SMOOTH * min(action_delta, 1.0)
        components["step_cost"] = self.R_STEP

        total = float(sum(components.values()))
        components["total"] = total
        self.last_reward_components = components
        return total, components

    def _physics(self, action: np.ndarray) -> None:
        ax, ay = float(action[0]) * self.ACCEL, float(action[1]) * self.ACCEL
        self.hit_boundary = False

        damping = 1.0 - self.FRICTION * self.DT
        self.vel_x = (self.vel_x + ax * self.DT) * damping
        self.vel_y = (self.vel_y + ay * self.DT) * damping

        speed = float(np.hypot(self.vel_x, self.vel_y))
        if speed > self.MAX_SPEED:
            self.vel_x *= self.MAX_SPEED / speed
            self.vel_y *= self.MAX_SPEED / speed

        self.pos_x += self.vel_x * self.DT
        self.pos_y += self.vel_y * self.DT

        for position_name, velocity_name in (("pos_x", "vel_x"), ("pos_y", "vel_y")):
            position = getattr(self, position_name)
            velocity = getattr(self, velocity_name)
            if position > self.BOUNDARY:
                setattr(self, position_name, self.BOUNDARY)
                setattr(self, velocity_name, -abs(velocity) * self.BOUNCE)
                self.hit_boundary = True
            elif position < -self.BOUNDARY:
                setattr(self, position_name, -self.BOUNDARY)
                setattr(self, velocity_name, abs(velocity) * self.BOUNCE)
                self.hit_boundary = True

    def get_coverage(self) -> float:
        return self.waypoint_idx / max(1, len(self.path_points) - 1)

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        keys: Iterable[str] = (
            "approach",
            "waypoint",
            "tracking_error",
            "off_curve",
            "backtrack",
            "boundary",
            "smooth",
            "step_cost",
            "completion",
            "partial80",
            "incomplete",
            "total",
        )
        return {key: 0.0 for key in keys}
