import os, re
import numpy as np
import matplotlib.pyplot as plt

base = "runs_sweep_WH"
# print(os.listdir(base))

widths_set, heights_set, records = set(), set(), []

for run in os.listdir(base):
    m = re.match(r"W_(?P<W>[0-9.]+)_H_(?P<H>[0-9.]+)", run)
    if not m:
        continue
    W = float(m.group("W"))
    H = float(m.group("H"))
    summary_path = os.path.join(base, run, "eval_summary.txt")
    if not os.path.exists(summary_path):
        continue

    collision_rate = None
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("collision_rate:"):
                collision_rate = float(line.split(":", 1)[1].strip())
                break
    if collision_rate is None:
        continue

    success = 1.0 - collision_rate
    widths_set.add(W)
    heights_set.add(H)
    records.append((W, H, success))

if not records:
    print("[heatmap] No eval_summary.txt found, skipping heatmap.")
    raise SystemExit(0)

widths = sorted(widths_set)
heights = sorted(heights_set)
Wn, Hn = len(widths), len(heights)

grid = np.zeros((Hn, Wn), dtype=float)
for Wv, Hv, succ in records:
    wi = widths.index(Wv)
    hi = heights.index(Hv)
    grid[hi, wi] = succ

np.save(os.path.join(base, "widths.npy"), np.array(widths, dtype=np.float32))
np.save(os.path.join(base, "heights.npy"), np.array(heights, dtype=np.float32))
np.save(os.path.join(base, "success_grid.npy"), grid)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)

ax.set_xticks(range(Wn))
ax.set_yticks(range(Hn))
ax.set_xticklabels([f"{w:.2f}" for w in widths])
ax.set_yticklabels([f"{h:.2f}" for h in heights])
ax.set_xlabel("Obstacle width (W)")
ax.set_ylabel("Obstacle height (H)")
ax.set_title("Success rate heatmap")

for i in range(Hn):
    for j in range(Wn):
        val = grid[i, j] * 100.0
        txt_color = "white" if val < 50 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                color=txt_color, fontsize=8)

fig.colorbar(im, ax=ax, label="Success rate")
plt.tight_layout()
heatmap_path = os.path.join(base, "success_heatmap.png")
plt.savefig(heatmap_path, dpi=160)
print(f"[heatmap] saved: {heatmap_path}")