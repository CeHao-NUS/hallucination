
import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from env import make_env
from model import FlowMatchingConfig, FlowMatchingModel


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(
    trajectories: np.ndarray,
    batch_size: int,
    val_fraction: float = 0.1,
    seed: int = 0,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    trajectories: (N, T, 2) float32
    """
    rng = np.random.default_rng(seed)
    N = trajectories.shape[0]
    idx = rng.permutation(N)
    n_val = int(np.floor(val_fraction * N))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train = torch.from_numpy(trajectories[train_idx])
    val = torch.from_numpy(trajectories[val_idx])

    train_ds = TensorDataset(train)
    val_ds = TensorDataset(val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_dl, val_dl


@torch.no_grad()
def evaluate(model: FlowMatchingModel, val_dl: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for (traj,) in val_dl:
        traj = traj.to(device)
        loss = model.flow_matching_loss(traj, reduction="mean")
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def save_losses(out_dir: str, history: Dict[str, list]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "loss_history.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i in range(len(history["train_loss"])):
            f.write(f"{i+1},{history['train_loss'][i]},{history['val_loss'][i]}\n")

    # PNG
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Flow-matching loss (MSE)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(png_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()

    # Output
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: runs/<timestamp>)")
    p.add_argument("--seed", type=int, default=0)

    # Env / data
    p.add_argument("--obstacle_width", type=float, default=1.0, help="Full obstacle width (this is your W knob)")
    p.add_argument("--obstacle_height", type=float, default=2.0)
    p.add_argument("--clearance", type=float, default=0.25)
    p.add_argument("--n_points", type=int, default=60)
    p.add_argument("--num_traj", type=int, default=5000)
    p.add_argument("--p_left", type=float, default=0.5)
    p.add_argument("--noise_std", type=float, default=0.04)
    p.add_argument("--val_fraction", type=float, default=0.1)

    # Model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--time_emb_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--activation", type=str, default="silu", choices=["silu", "tanh", "relu"])
    p.add_argument("--spectral_norm", type=int, default=0, help="1 to enable spectral norm (Lipschitz control proxy)")
    p.add_argument("--sampler", type=str, default="euler", choices=["euler", "rk4"])

    # Training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs)")
    p.add_argument("--min_delta", type=float, default=1e-5, help="Min val improvement to reset patience")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)

    args = p.parse_args()

    set_seed(args.seed)

    # Output directory
    if args.out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("runs", ts)
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # Build env + generate dataset
    env = make_env(
        obstacle_width=args.obstacle_width,
        obstacle_height=args.obstacle_height,
        clearance=args.clearance,
    )
    data = env.generate_dataset(
        num_traj=args.num_traj,
        n_points=args.n_points,
        p_left=args.p_left,
        noise_std=args.noise_std,
        seed=args.seed,
        ensure_collision_free=True,
        return_modes=True,
    )
    trajectories = data["trajectories"]  # (N,T,2)

    # Optional sanity check collision rate (should be ~0)
    col_rate = np.mean([env.collides(tr) for tr in trajectories])
    print(f"[train] dataset collision rate: {col_rate*100:.2f}% (expected ~0%)")

    # Save config snapshot
    config_path = os.path.join(out_dir, "run_config.json")
    import json as _json
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(_json.dumps({"args": vars(args)}, indent=2))

    # Dataloaders
    train_dl, val_dl = make_dataloaders(
        trajectories=trajectories,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # Model
    cfg = FlowMatchingConfig(
        n_points=args.n_points,
        point_dim=2,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        time_emb_dim=args.time_emb_dim,
        cond_dim=0,
        dropout=args.dropout,
        activation=args.activation,
        spectral_norm=bool(args.spectral_norm),
        sampler=args.sampler,
    )
    model = FlowMatchingModel(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.max_epochs, 1))

    # Training loop with early stopping
    best_val = float("inf")
    best_epoch = -1
    patience_left = args.patience
    history = {"train_loss": [], "val_loss": []}

    ckpt_path = os.path.join(out_dir, "checkpoint_best.pt")

    print("[train] starting...")
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []

        for (traj,) in train_dl:
            traj = traj.to(device)
            loss = model.flow_matching_loss(traj, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        scheduler.step()

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = evaluate(model, val_dl, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = (best_val - val_loss) > args.min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience_left = args.patience
            model.save(ckpt_path)
        else:
            patience_left -= 1

        if epoch % 10 == 0 or epoch == 1:
            lr = opt.param_groups[0]["lr"]
            print(
                f"[epoch {epoch:04d}] train={train_loss:.6f}  val={val_loss:.6f}  "
                f"best={best_val:.6f}@{best_epoch}  patience={patience_left}  lr={lr:.2e}"
            )

        if epoch % 10 == 0 or epoch == args.max_epochs:
            save_losses(out_dir, history)

        if patience_left <= 0:
            print(f"[train] early stopping at epoch {epoch} (best epoch {best_epoch}, best val {best_val:.6f})")
            break

    save_losses(out_dir, history)

    print(f"[train] done. best checkpoint: {ckpt_path}")
    print(f"[train] loss curve: {os.path.join(out_dir, 'loss_curve.png')}")
    print(f"[train] history csv: {os.path.join(out_dir, 'loss_history.csv')}")


if __name__ == "__main__":
    main()
