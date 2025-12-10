from pathlib import Path

"""
env.py

A minimal 2D obstacle-avoidance environment for validating the "topological barrier":
- Start at bottom, goal at top (by default).
- A rectangular obstacle block sits in the middle.
- Safe demonstrations come in two modes: going around the obstacle on the LEFT or RIGHT.

This file focuses on:
1) Defining the environment geometry (rectangle obstacle).
2) Generating collision-free trajectory datasets for downstream training.

No simulator required; trajectories are polylines in R^2.

Author: (you)
"""


from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

Mode = Literal["left", "right"]


@dataclass(frozen=True)
class RectObstacle:
    """Axis-aligned rectangle obstacle."""
    cx: float = 0.0
    cy: float = 0.0
    width: float = 1.0   # full width
    height: float = 2.0  # full height

    @property
    def xmin(self) -> float:
        return self.cx - self.width / 2.0

    @property
    def xmax(self) -> float:
        return self.cx + self.width / 2.0

    @property
    def ymin(self) -> float:
        return self.cy - self.height / 2.0

    @property
    def ymax(self) -> float:
        return self.cy + self.height / 2.0

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask indicating whether each point is inside the obstacle.

        points: (N,2) array.
        """
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must have shape (N,2); got {pts.shape}")
        x, y = pts[:, 0], pts[:, 1]
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)


@dataclass(frozen=True)
class EnvConfig:
    """
    Configuration for the obstacle-avoidance toy environment.

    Coordinates are 2D (x,y). By default:
      start = (0, -2.5)
      goal  = (0,  2.5)
      obstacle centered at (0,0)
    """
    start: Tuple[float, float] = (0.0, -2.5)
    goal: Tuple[float, float] = (0.0, 2.5)
    obstacle: RectObstacle = RectObstacle(cx=0.0, cy=0.0, width=1.0, height=2.0)

    # Optional bounding box for plotting / sanity checks
    xlim: Tuple[float, float] = (-3.0, 3.0)
    ylim: Tuple[float, float] = (-3.0, 3.0)

    # How tightly the trajectory squeezes around the obstacle (>= 0).
    # Larger => wider berth around the obstacle.
    clearance: float = 0.25


class ObstacleAvoidanceEnv:
    """
    A simple geometric environment with two (or more) modes around a rectangular obstacle.

    The "ground truth" safe trajectories are generated to be collision-free with a chosen clearance.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    @property
    def start(self) -> np.ndarray:
        return np.array(self.cfg.start, dtype=np.float32)

    @property
    def goal(self) -> np.ndarray:
        return np.array(self.cfg.goal, dtype=np.float32)

    @property
    def obstacle(self) -> RectObstacle:
        return self.cfg.obstacle

    def collides(self, traj: np.ndarray) -> bool:
        """
        Collision check for a polyline trajectory: returns True if ANY point lies inside obstacle.

        traj: (T,2)
        """
        return bool(np.any(self.obstacle.contains(traj)))

    def _min_sin_over_obstacle_band(self, y: np.ndarray) -> float:
        """
        Helper: For the default parametric path x(t) = A * sin(pi t),
        find min(sin(pi t)) over t values corresponding to y in obstacle vertical range.
        """
        sy, gy = float(self.start[1]), float(self.goal[1])
        if gy == sy:
            return 0.0
        t = (y - sy) / (gy - sy)
        t = np.clip(t, 0.0, 1.0)
        s = np.sin(np.pi * t)
        return float(np.min(s))

    def sample_clean_path(
        self,
        mode: Mode,
        n_points: int = 50,
        clearance: Optional[float] = None,
        curve: Literal["sin"] = "sin",
    ) -> np.ndarray:
        """
        Generate a single collision-free trajectory in the specified mode.

        The default curve is a "side-bulge" path:
          y(t) = linear interpolation from start_y to goal_y
          x(t) = sign * A * sin(pi t)
        which starts and ends at x=0 and reaches maximum lateral deviation at mid-time.

        We choose A large enough so that throughout the obstacle's y-range,
        |x(t)| stays outside the obstacle's x-range by at least `clearance`.

        Returns:
          traj: (n_points, 2) float32
        """
        if n_points < 2:
            raise ValueError("n_points must be >= 2")

        sign = -1.0 if mode == "left" else 1.0
        clr = self.cfg.clearance if clearance is None else float(clearance)

        s = self.start
        g = self.goal

        t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
        y = s[1] + t * (g[1] - s[1])

        # Obstacle half-width/height
        half_w = self.obstacle.width / 2.0
        half_h = self.obstacle.height / 2.0

        # We need |x(t)| >= half_w + clr whenever y(t) in [cy-half_h, cy+half_h]
        # For x(t)=A*sin(pi t), the minimum sin(pi t) within that y-band determines required A.
        y_band = np.array([self.obstacle.cy - half_h, self.obstacle.cy + half_h], dtype=np.float32)
        min_sin = self._min_sin_over_obstacle_band(y_band)
        # If obstacle band is outside the path's y-range or degeneracy occurs:
        min_sin = max(min_sin, 1e-3)

        required_A = (half_w + clr) / min_sin

        if curve != "sin":
            raise ValueError(f"Unsupported curve='{curve}'. Only 'sin' is implemented.")

        x = s[0] + (g[0] - s[0]) * t  # usually 0 -> 0
        x = x + sign * required_A * np.sin(np.pi * t)

        traj = np.stack([x, y], axis=-1).astype(np.float32)

        # Sanity: endpoints exact
        traj[0] = s
        traj[-1] = g

        # Safety check: should not collide (unless clearance is negative or geometry is weird)
        if self.collides(traj):
            raise RuntimeError(
                "Generated 'clean' path collides with obstacle. "
                "Try increasing clearance or expanding workspace."
            )

        return traj

    def sample_noisy_path(
        self,
        mode: Mode,
        n_points: int = 50,
        noise_std: float = 0.03,
        clearance: Optional[float] = None,
        max_tries: int = 50,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate a (typically) collision-free trajectory with small noise, by rejection sampling.

        Noise is added to intermediate points only; start/goal remain fixed.
        If ensure collision-free examples, keep trying until collision-free or max_tries exceeded.
        """
        rng = np.random.default_rng() if rng is None else rng
        base = self.sample_clean_path(mode=mode, n_points=n_points, clearance=clearance)

        for _ in range(max_tries):
            traj = base.copy()
            if n_points > 2:
                traj[1:-1] += rng.normal(0.0, noise_std, size=(n_points - 2, 2)).astype(np.float32)
            traj[0] = self.start
            traj[-1] = self.goal
            if not self.collides(traj):
                return traj

        # If we can't find a collision-free noisy sample, fall back to base.
        return base

    def generate_dataset(
        self,
        num_traj: int,
        n_points: int = 50,
        p_left: float = 0.5,
        noise_std: float = 0.03,
        clearance: Optional[float] = None,
        seed: Optional[int] = None,
        ensure_collision_free: bool = True,
        max_tries: int = 50,
        return_modes: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a dataset of trajectories.

        Args:
          num_traj: number of trajectories
          n_points: points per trajectory
          p_left: probability of sampling left-mode (right-mode has prob 1 - p_left)
          noise_std: gaussian noise applied to intermediate points
          clearance: override environment clearance for dataset generation
          seed: RNG seed
          ensure_collision_free: if True, use rejection sampling for noise; otherwise allow collisions
          max_tries: max rejection tries per sample if ensure_collision_free
          return_modes: whether to include mode labels

        Returns dict:
          'trajectories': (N, T, 2) float32
          'modes': (N,) int64   (0=left, 1=right)  [if return_modes]
        """
        if not (0.0 <= p_left <= 1.0):
            raise ValueError("p_left must be in [0,1]")

        rng = np.random.default_rng(seed)
        trajs = np.zeros((num_traj, n_points, 2), dtype=np.float32)
        modes = np.zeros((num_traj,), dtype=np.int64)

        for i in range(num_traj):
            is_left = rng.random() < p_left
            mode: Mode = "left" if is_left else "right"
            modes[i] = 0 if is_left else 1

            if noise_std <= 0:
                traj = self.sample_clean_path(mode=mode, n_points=n_points, clearance=clearance)
            else:
                if ensure_collision_free:
                    traj = self.sample_noisy_path(
                        mode=mode,
                        n_points=n_points,
                        noise_std=noise_std,
                        clearance=clearance,
                        max_tries=max_tries,
                        rng=rng,
                    )
                else:
                    base = self.sample_clean_path(mode=mode, n_points=n_points, clearance=clearance)
                    traj = base.copy()
                    if n_points > 2:
                        traj[1:-1] += rng.normal(0.0, noise_std, size=(n_points - 2, 2)).astype(np.float32)
                    traj[0] = self.start
                    traj[-1] = self.goal

            trajs[i] = traj

        out: Dict[str, np.ndarray] = {"trajectories": trajs}
        if return_modes:
            out["modes"] = modes
        return out

    def plot(
        self,
        trajectories: np.ndarray,
        show: bool = True,
        title: Optional[str] = None,
        color_by_collision: bool = True,
        obstacle_alpha: float = 0.35,
    ) -> None:
        """
        Quick visualization helper (requires matplotlib).

        trajectories: (N,T,2) or (T,2)
        """
        import matplotlib.pyplot as plt  # local import

        trajs = np.asarray(trajectories)
        if trajs.ndim == 2:
            trajs = trajs[None, ...]
        if trajs.ndim != 3 or trajs.shape[-1] != 2:
            raise ValueError(f"trajectories must have shape (N,T,2) or (T,2); got {trajs.shape}")

        plt.figure(figsize=(6, 6))
        # obstacle
        ob = self.obstacle
        rect = plt.Rectangle((ob.xmin, ob.ymin), ob.width, ob.height, color="gray", alpha=obstacle_alpha)
        plt.gca().add_patch(rect)

        for traj in trajs:
            c = "red" if (color_by_collision and self.collides(traj)) else "blue"
            plt.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.25)

        plt.scatter([self.start[0]], [self.start[1]], c="green", s=60, label="Start")
        plt.scatter([self.goal[0]], [self.goal[1]], c="purple", s=60, marker="*", label="Goal")

        plt.xlim(*self.cfg.xlim)
        plt.ylim(*self.cfg.ylim)
        plt.gca().set_aspect("equal", "box")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if title:
            plt.title(title)
        if show:
            # plt.show()
            # save locally
            plt.savefig("obstacle_avoidance_plot.png")

def make_env(
    obstacle_width: float = 1.0,
    obstacle_height: float = 2.0,
    clearance: float = 0.25,
    start: Tuple[float, float] = (0.0, -2.5),
    goal: Tuple[float, float] = (0.0, 2.5),
    xlim: Tuple[float, float] = (-3.0, 3.0),
    ylim: Tuple[float, float] = (-3.0, 3.0),
) -> ObstacleAvoidanceEnv:
    """
    Convenience constructor so you can pass parameters as variables.
    This is where you'll vary W (obstacle width) in experiments.
    """
    cfg = EnvConfig(
        start=start,
        goal=goal,
        obstacle=RectObstacle(cx=0.0, cy=0.0, width=float(obstacle_width), height=float(obstacle_height)),
        xlim=xlim,
        ylim=ylim,
        clearance=float(clearance),
    )
    return ObstacleAvoidanceEnv(cfg)


if __name__ == "__main__":
    # Minimal smoke test
    env = make_env(obstacle_width=1.0, obstacle_height=2.0, clearance=0.25)
    data = env.generate_dataset(num_traj=200, n_points=60, p_left=0.5, noise_std=0.04, seed=0)
    trajs = data["trajectories"]
    collision_rate = np.mean([env.collides(tr) for tr in trajs])
    print(f"Generated {len(trajs)} trajectories. Collision rate (should be ~0): {collision_rate:.3f}")
    env.plot(trajs[:200], title="Obstacle Avoidance Dataset (Left/Right Modes)")


# path = Path("/mnt/data/env.py")
# path.write_text(env_code, encoding="utf-8")
# str(path)


