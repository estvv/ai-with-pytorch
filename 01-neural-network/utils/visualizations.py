"""
python visualizations.py  ->  training.png
Progression de l'entraînement : données → frontières → convergence.
Importe le modèle entraîné depuis main.py.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_circles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import NeuralNet

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
X_np, y_np = make_circles(n_samples=1000, noise=0.1, factor=0.4, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

xs_0  = X_np[y_np == 0, 0]; ys_c0 = X_np[y_np == 0, 1]
xs_1  = X_np[y_np == 1, 0]; ys_c1 = X_np[y_np == 1, 1]


# ─────────────────────────────────────────────
# ENTRAÎNEMENT — snapshots à différents epochs
# ─────────────────────────────────────────────
def train_with_history(n_hidden=16, lr=0.01, epochs=500):
    model     = NeuralNet(n_hidden)
    loss_fn   = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses    = []
    snapshots = {}  # epoch → state_dict

    for epoch in range(epochs):
        y_hat = model(X)
        loss  = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch in (0, 24, 99, epochs - 1):
            snapshots[epoch] = {k: v.clone() for k, v in model.state_dict().items()}

    return losses, snapshots, n_hidden


losses, snapshots, n_hidden = train_with_history()
snap_epochs = [0, 24, 99, 499]
snap_colors = ["#ffd166", "#f4a261", "#e76f51", "#00ff99"]
snap_labels = ["epoch 1", "epoch 25", "epoch 100", "epoch 500 (final)"]


# ─────────────────────────────────────────────
# DÉCISION BOUNDARY
# ─────────────────────────────────────────────
def boundary(state_dict, res=80):
    model = NeuralNet(n_hidden)
    model.load_state_dict(state_dict)
    model.eval()

    grid  = np.linspace(-3.5, 3.5, res)
    xx, yy = np.meshgrid(grid, grid)
    pts   = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        zz = model(pts).numpy().reshape(res, res)

    return zz


# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
BG, AX_BG, GRID_COL = "#0f0f1a", "#13132b", "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5)


# ─────────────────────────────────────────────
# FIGURE — 6 panels
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(17, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("Neural Network (PyTorch) — du problème à la convergence",
             color="white", fontsize=14, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
EXT = [-3.5, 3.5, -3.5, 3.5]


# Panel 1 — données brutes
ax1 = fig.add_subplot(gs[0, 0])
_style(ax1, "Step 1 — Les données\nNon-linéairement séparables")
ax1.scatter(xs_0, ys_c0, color="#7ec8e3", s=22, alpha=0.85, zorder=3, label="classe 0 (intérieur)")
ax1.scatter(xs_1, ys_c1, color="#ff6b6b", s=22, alpha=0.85, zorder=3, label="classe 1 (extérieur)")
ax1.set_aspect("equal")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7, loc="upper right")


# Panel 2 — pourquoi une droite échoue
ax2 = fig.add_subplot(gs[0, 1])
_style(ax2, "Step 2 — Une droite ne peut pas séparer\nIl faut déformer l'espace")
ax2.scatter(xs_0, ys_c0, color="#7ec8e3", s=22, alpha=0.7, zorder=3)
ax2.scatter(xs_1, ys_c1, color="#ff6b6b", s=22, alpha=0.7, zorder=3)
for slope in [0.0, 0.8, -0.8, 2.0]:
    ax2.axline((0, 0), slope=slope, color="#ffd166", lw=1.2, alpha=0.4, linestyle="--")
ax2.text(0, 2.8, "n'importe quelle\ndroite échoue", color="#ffd166", fontsize=8, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1500", edgecolor="#ffd166"))
ax2.set_xlim(-3.5, 3.5); ax2.set_ylim(-3.5, 3.5); ax2.set_aspect("equal")


# Panel 3 — frontière epoch 1
ax3 = fig.add_subplot(gs[0, 2])
_style(ax3, "Step 3 — Epoch 1\nPoids aléatoires, frontière aléatoire")
zz = boundary(snapshots[snap_epochs[0]])
ax3.imshow(zz, origin="lower", aspect="auto", extent=EXT,
           cmap="RdBu_r", vmin=0, vmax=1, alpha=0.65)
ax3.scatter(xs_0, ys_c0, color="#7ec8e3", s=18, alpha=0.9, zorder=3)
ax3.scatter(xs_1, ys_c1, color="#ff6b6b", s=18, alpha=0.9, zorder=3)
ax3.set_aspect("equal")


# Panel 4 — frontière epoch 25
ax4 = fig.add_subplot(gs[1, 0])
_style(ax4, "Step 4 — Epoch 25\nLa frontière commence à prendre forme")
zz = boundary(snapshots[snap_epochs[1]])
ax4.imshow(zz, origin="lower", aspect="auto", extent=EXT,
           cmap="RdBu_r", vmin=0, vmax=1, alpha=0.65)
ax4.scatter(xs_0, ys_c0, color="#7ec8e3", s=18, alpha=0.9, zorder=3)
ax4.scatter(xs_1, ys_c1, color="#ff6b6b", s=18, alpha=0.9, zorder=3)
ax4.set_aspect("equal")


# Panel 5 — frontière finale
ax5 = fig.add_subplot(gs[1, 1])
_style(ax5, "Step 5 — Epoch 500 (final)\nLes cercles sont séparés")
zz = boundary(snapshots[snap_epochs[3]])
ax5.imshow(zz, origin="lower", aspect="auto", extent=EXT,
           cmap="RdBu_r", vmin=0, vmax=1, alpha=0.65)
ax5.scatter(xs_0, ys_c0, color="#7ec8e3", s=18, alpha=0.9, zorder=3)
ax5.scatter(xs_1, ys_c1, color="#ff6b6b", s=18, alpha=0.9, zorder=3)
ax5.set_aspect("equal")


# Panel 6 — courbe de loss
ax6 = fig.add_subplot(gs[1, 2])
_style(ax6, "Step 6 — Loss par epoch\nConverge vers le minimum")
ax6.plot(range(len(losses)), losses, color="#7ec8e3", lw=2)

for ep, col in zip(snap_epochs, snap_colors):
    ax6.axvline(ep, color=col, linestyle=":", lw=1, alpha=0.7)
    ax6.plot(ep, losses[ep], "o", color=col, markersize=6, zorder=5)

ax6.text(len(losses) * 0.4, losses[0] * 0.75,
         f"loss finale : {losses[-1]:.4f}",
         color="white", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333355"))
ax6.set_xlabel("epoch", color="#888888", fontsize=8)
ax6.set_ylabel("BCE loss", color="#888888", fontsize=8)


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "training.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")
