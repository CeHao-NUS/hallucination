from pathlib import Path

import argparse
import os
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from env import make_env
from model import FlowMatchingModel


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_trajectories(
    env,
    trajs: np.ndarray,
    collisions: np.ndarray,
    out_path: str,
    max_plot: int = 400,
    obstacle_alpha: float = 0.35,
    title: Optional[str] = None,
) -> None:
    """
    trajs: (N,T,2)
    collisions: (N,) bool
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    N = trajs.shape[0]
    n_plot = min(N, max_plot)
    idx = np.arange(N)
    if N > n_plot:
        idx = np.random.choice(N, size=n_plot, replace=False)

    plt.figure(figsize=(6, 6))

    # obstacle
    ob = env.obstacle
    rect = plt.Rectangle((ob.xmin, ob.ymin), ob.width, ob.height, color="gray", alpha=obstacle_alpha)
    plt.gca().add_patch(rect)

    # trajectories
    for i in idx:
        tr = trajs[i]
        if collisions[i]:
            plt.plot(tr[:, 0], tr[:, 1], color="red", alpha=0.35, linewidth=1.0)
        else:
            plt.plot(tr[:, 0], tr[:, 1], color="blue", alpha=0.22, linewidth=1.0)

    # start/goal
    s = env.start
    g = env.goal
    plt.scatter([s[0]], [s[1]], c="green", s=70, label="Start")
    plt.scatter([g[0]], [g[1]], c="purple", s=90, marker="*", label="Goal")

    plt.xlim(*env.cfg.xlim)
    plt.ylim(*env.cfg.ylim)
    plt.gca().set_aspect("equal", "box")
    plt.grid(True, alpha=0.3)

    succ = int(np.sum(~collisions))
    fail = int(np.sum(collisions))
    rate = float(np.mean(collisions)) if len(collisions) else float("nan")

    if title is None:
        title = f"Samples: {N} | Success: {succ} | Collide: {fail} ({rate*100:.2f}%)"
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    p = argparse.ArgumentParser()

    # Inputs/outputs
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint_best.pt")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: same dir as ckpt)")
    p.add_argument("--seed", type=int, default=0)

    # Env (must match training for meaningful collision rate)
    p.add_argument("--obstacle_width", type=float, default=1.0)
    p.add_argument("--obstacle_height", type=float, default=2.0)
    p.add_argument("--clearance", type=float, default=0.25)
    p.add_argument("--xlim", type=float, nargs=2, default=(-3.0, 3.0))
    p.add_argument("--ylim", type=float, nargs=2, default=(-3.0, 3.0))
    p.add_argument("--start", type=float, nargs=2, default=(0.0, -2.5))
    p.add_argument("--goal", type=float, nargs=2, default=(0.0, 2.5))

    # Sampling
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--n_steps", type=int, default=80)
    p.add_argument("--max_plot", type=int, default=400)
    p.add_argument("--save_npy", type=int, default=1, help="1 to save generated trajectories as .npy")

    args = p.parse_args()

    set_seed(args.seed)

    # Output dir
    if args.out_dir is None:
        out_dir = os.path.dirname(args.ckpt) or "."
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    # Env
    env = make_env(
        obstacle_width=args.obstacle_width,
        obstacle_height=args.obstacle_height,
        clearance=args.clearance,
        start=tuple(args.start),
        goal=tuple(args.goal),
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
    )

    # Load model
    model = FlowMatchingModel.load(args.ckpt, map_location=str(device))
    model = model.to(device)
    model.eval()

    # Sample
    with torch.no_grad():
        trajs_t = model.sample(n_samples=args.n_samples, n_steps=args.n_steps, device=device)
    trajs = trajs_t.detach().cpu().numpy()

    # Collision check
    collisions = np.array([env.collides(tr) for tr in trajs], dtype=bool)
    rate = float(np.mean(collisions)) if len(collisions) else float("nan")
    print(f"[eval] collision/hallucination rate: {rate*100:.2f}% ({int(collisions.sum())}/{len(collisions)})")

    # Save outputs
    plot_path = os.path.join(out_dir, "samples_collision_plot.png")
    plot_trajectories(env, trajs, collisions, out_path=plot_path, max_plot=args.max_plot)

    if args.save_npy:
        np.save(os.path.join(out_dir, "samples_trajs.npy"), trajs)
        np.save(os.path.join(out_dir, "samples_collisions.npy"), collisions.astype(np.int8))

    # Also save a small text summary
    with open(os.path.join(out_dir, "eval_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"n_samples: {args.n_samples}\n")
        f.write(f"n_steps: {args.n_steps}\n")
        f.write(f"collision_rate: {rate}\n")
        f.write(f"collisions: {int(collisions.sum())}/{len(collisions)}\n")
        f.write(f"obstacle_width: {args.obstacle_width}\n")
        f.write(f"obstacle_height: {args.obstacle_height}\n")
        f.write(f"clearance: {args.clearance}\n")

    print(f"[eval] wrote plot: {plot_path}")
    if args.save_npy:
        print(f"[eval] wrote npy: {os.path.join(out_dir, 'samples_trajs.npy')}")
    print(f"[eval] wrote summary: {os.path.join(out_dir, 'eval_summary.txt')}")


if __name__ == "__main__":
    main()
