from pathlib import Path



from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Literal

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def _time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal time embedding used in diffusion/flows.
    t: (B,) in [0,1]
    returns: (B, dim)
    """
    if t.ndim != 1:
        t = t.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device, dtype=t.dtype)], dim=-1)
    return emb


def _maybe_apply_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    """
    Apply spectral normalization to Linear layers if enabled.
    Uses torch.nn.utils.parametrizations.spectral_norm when available, otherwise falls back.
    """
    if not enabled:
        return module

    try:
        from torch.nn.utils.parametrizations import spectral_norm as param_spectral_norm  # type: ignore
        Spectral = param_spectral_norm
    except Exception:
        # Older PyTorch fallback
        from torch.nn.utils import spectral_norm as Spectral  # type: ignore

    for name, layer in module.named_modules():
        if isinstance(layer, nn.Linear):
            try:
                Spectral(layer)
            except Exception:
                # Some layers might already be wrapped
                pass
    return module


# -----------------------------
# Model
# -----------------------------

@dataclass
class FlowMatchingConfig:
    # Trajectory shape
    n_points: int = 50           # T
    point_dim: int = 2           # 2D (x,y)
    # Latent / data dimension is D = n_points * point_dim

    # Network
    hidden_dim: int = 256
    depth: int = 4
    time_emb_dim: int = 64
    cond_dim: int = 0            # set >0 if you want conditioning
    dropout: float = 0.0
    activation: Literal["silu", "tanh", "relu"] = "silu"

    # Lipschitz control
    spectral_norm: bool = False

    # Sampling
    sampler: Literal["euler", "rk4"] = "euler"


class MLPVelocityField(nn.Module):
    """
    v_theta(x, t, cond) -> velocity in R^D
    Where x is flattened trajectory vector.
    """

    def __init__(self, dim: int, cfg: FlowMatchingConfig):
        super().__init__()
        self.dim = dim
        self.cfg = cfg

        act: nn.Module
        if cfg.activation == "silu":
            act = nn.SiLU()
        elif cfg.activation == "tanh":
            act = nn.Tanh()
        elif cfg.activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {cfg.activation}")

        in_dim = dim + cfg.time_emb_dim + (cfg.cond_dim if cfg.cond_dim > 0 else 0)

        layers = []
        h = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, h))
        layers.append(act)
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))

        for _ in range(cfg.depth - 1):
            layers.append(nn.Linear(h, h))
            layers.append(act)
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))

        layers.append(nn.Linear(h, dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, D)
        t: (B,) in [0,1]
        cond: (B, cond_dim) optional
        """
        te = _time_embedding(t, self.cfg.time_emb_dim)
        if self.cfg.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required but cfg.cond_dim > 0")
            if cond.ndim != 2 or cond.shape[1] != self.cfg.cond_dim:
                raise ValueError(f"cond must have shape (B,{self.cfg.cond_dim}); got {tuple(cond.shape)}")
            inp = torch.cat([x, te, cond], dim=-1)
        else:
            inp = torch.cat([x, te], dim=-1)
        return self.net(inp)


class FlowMatchingModel(nn.Module):
    """
    High-level wrapper:
    - Holds a velocity field v_theta
    - Implements flow matching loss
    - Implements ODE sampling (Euler / RK4)
    - Save / load utilities
    """

    def __init__(self, cfg: FlowMatchingConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.n_points * cfg.point_dim

        vf = MLPVelocityField(dim=self.dim, cfg=cfg)
        self.vf = _maybe_apply_spectral_norm(vf, cfg.spectral_norm)

    # -------- trajectory shape helpers --------
    def flatten_traj(self, traj: torch.Tensor) -> torch.Tensor:
        """
        traj: (B, T, 2) -> (B, D)
        """
        if traj.ndim != 3 or traj.shape[1] != self.cfg.n_points or traj.shape[2] != self.cfg.point_dim:
            raise ValueError(
                f"traj must have shape (B,{self.cfg.n_points},{self.cfg.point_dim}); got {tuple(traj.shape)}"
            )
        return traj.reshape(traj.shape[0], -1)

    def unflatten_traj(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) -> (B, T, 2)
        """
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (B,{self.dim}); got {tuple(x.shape)}")
        return x.view(x.shape[0], self.cfg.n_points, self.cfg.point_dim)

    # -------- core network --------
    def velocity(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.vf(x, t, cond)

    # -------- training objective --------
    @torch.no_grad()
    def sample_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(batch_size, self.dim, device=device, dtype=dtype)

    def flow_matching_loss(
        self,
        traj_data: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """
        Flow matching (rectified flow) loss.

        traj_data: (B, T, 2)
        cond: (B, cond_dim) optional

        Returns: scalar loss
        """
        x1 = self.flatten_traj(traj_data)
        B = x1.shape[0]
        device, dtype = x1.device, x1.dtype

        # Sample base noise and time
        x0 = self.sample_noise(B, device=device, dtype=dtype)
        t = torch.rand(B, device=device, dtype=dtype)

        # Interpolate and target
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        ut = x1 - x0

        # Predict velocity
        vt = self.velocity(xt, t, cond)

        loss = F.mse_loss(vt, ut, reduction=reduction)
        return loss

    # -------- sampling / ODE solve --------
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_steps: int = 50,
        cond: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        z: Optional[torch.Tensor] = None,
        return_flat: bool = False,
    ) -> torch.Tensor:
        """
        Generate trajectories by integrating dx/dt = v_theta(x,t,cond) from t=0->1.

        Args:
          n_samples: batch size
          n_steps: number of solver steps
          cond: (B, cond_dim) or None
          z: optional initial noise (B,D). If provided, overrides random sampling.
          sampler: uses cfg.sampler ("euler" or "rk4")
          return_flat: if True, return (B,D), else return (B,T,2)

        Returns:
          trajectories: (B,T,2) float tensor (or flat if return_flat)
        """
        device = device or next(self.parameters()).device
        if z is None:
            x = torch.randn(n_samples, self.dim, device=device, dtype=dtype)
        else:
            if z.shape != (n_samples, self.dim):
                raise ValueError(f"z must have shape (B,{self.dim}); got {tuple(z.shape)}")
            x = z.to(device=device, dtype=dtype)

        # Validate cond
        if self.cfg.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required but cfg.cond_dim > 0")
            if cond.shape[0] != n_samples or cond.shape[1] != self.cfg.cond_dim:
                raise ValueError(f"cond must have shape (B,{self.cfg.cond_dim}); got {tuple(cond.shape)}")
            cond = cond.to(device=device, dtype=dtype)

        dt = 1.0 / float(n_steps)
        sampler = self.cfg.sampler

        def f(xi: torch.Tensor, ti: torch.Tensor) -> torch.Tensor:
            return self.velocity(xi, ti, cond)

        for k in range(n_steps):
            t0 = torch.full((n_samples,), k * dt, device=device, dtype=dtype)

            if sampler == "euler":
                x = x + dt * f(x, t0)
            elif sampler == "rk4":
                # classical RK4 on [t, t+dt]
                k1 = f(x, t0)
                k2 = f(x + 0.5 * dt * k1, t0 + 0.5 * dt)
                k3 = f(x + 0.5 * dt * k2, t0 + 0.5 * dt)
                k4 = f(x + dt * k3, t0 + dt)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

        if return_flat:
            return x
        return self.unflatten_traj(x)

    # -------- save/load --------
    def save(self, path: str) -> None:
        """
        Save model weights + config to a single .pt file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, map_location: Optional[str] = None) -> "FlowMatchingModel":
        """
        Load model from a .pt file created by save().
        """
        payload = torch.load(path, map_location=map_location)
        cfg_dict = payload["config"]
        cfg = FlowMatchingConfig(**cfg_dict)
        model = FlowMatchingModel(cfg)
        model.load_state_dict(payload["state_dict"])
        return model

    def to_config_json(self) -> str:
        return json.dumps(asdict(self.cfg), indent=2)

    # -------- optional: Lipschitz proxy --------
    @torch.no_grad()
    def spectral_norm_product_upper_bound(self) -> Optional[float]:
        """
        Very rough upper bound proxy for global Lipschitz of the MLP under spectral norm.
        If spectral normalization is enabled and successfully applied, Linear layers have
        spectral norm ~1 (up to parametrization accuracy). In that case, the product is ~1.
        If not enabled, returns None.

        This is not a rigorous global Lipschitz bound for the full velocity field + embeddings,
        but it's a handy sanity metric.
        """
        if not self.cfg.spectral_norm:
            return None
        prod = 1.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # With parametrizations, weight is not a plain Parameter; norm should be ~1.
                w = m.weight
                sn = torch.linalg.matrix_norm(w, ord=2).item()
                prod *= sn
        return float(prod)


if __name__ == "__main__":
    # Minimal smoke test: forward + loss + sample
    cfg = FlowMatchingConfig(n_points=60, hidden_dim=128, depth=3, spectral_norm=False, sampler="euler")
    model = FlowMatchingModel(cfg)

    B = 8
    traj = torch.randn(B, cfg.n_points, cfg.point_dim)
    loss = model.flow_matching_loss(traj)
    print("loss:", float(loss))

    samples = model.sample(n_samples=4, n_steps=20)
    print("samples:", samples.shape)

    tmp_path = "flow_matching_model.pt"
    model.save(tmp_path)
    model2 = FlowMatchingModel.load(tmp_path, map_location="cpu")
    print("loaded OK:", isinstance(model2, FlowMatchingModel))
