"""
Cas concret : détection d'anomalies réseau.
Un serveur est sain si sa charge CPU et sa latence sont proches des valeurs normales.
Il est en anomalie si l'une des deux est trop extrême.

Run:  python example.py
Output: example.png
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

BG, AX_BG, GRID = "#0f0f1a", "#13132b", "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.6)


# ─────────────────────────────────────────────
# MAPPING  espace normalisé  <->  unités réelles
#
# Valeurs normales : CPU 40%, latence 80ms
# 1 unité normalisée = 20%CPU / 40ms
# ─────────────────────────────────────────────
CPU_CENTER, CPU_SCALE     = 40.0, 20.0   # %
LAT_CENTER, LAT_SCALE     = 80.0, 40.0   # ms

def to_norm(cpu, lat):
    return [(cpu - CPU_CENTER) / CPU_SCALE,
            (lat - LAT_CENTER) / LAT_SCALE]

def to_real_cpu(x_norm): return CPU_CENTER + x_norm * CPU_SCALE
def to_real_lat(y_norm): return LAT_CENTER + y_norm * LAT_SCALE


# ─────────────────────────────────────────────
# DONNÉES
# Sain     (classe 0) : r faible — proche des valeurs normales
# Anomalie (classe 1) : r grand  — trop loin des valeurs normales
# ─────────────────────────────────────────────
X_np, y_np = make_circles(n_samples=1000, noise=0.1, factor=0.4, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)


# ─────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────
model     = NeuralNet(n_hidden=16)
loss_fn   = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses    = []

for _ in range(500):
    y_hat = model(X)
    loss  = loss_fn(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

model.eval()


# ─────────────────────────────────────────────
# DÉCISION BOUNDARY
# ─────────────────────────────────────────────
res  = 80
grid = np.linspace(-3.5, 3.5, res)
xx, yy = np.meshgrid(grid, grid)
pts  = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    ZZ = model(pts).numpy().reshape(res, res)

cpu_grid = [to_real_cpu(v) for v in grid]
lat_grid = [to_real_lat(v) for v in grid]

cpu_0 = [to_real_cpu(X_np[i, 0]) for i in range(len(X_np)) if y_np[i] == 0]
lat_0 = [to_real_lat(X_np[i, 1]) for i in range(len(X_np)) if y_np[i] == 0]
cpu_1 = [to_real_cpu(X_np[i, 0]) for i in range(len(X_np)) if y_np[i] == 1]
lat_1 = [to_real_lat(X_np[i, 1]) for i in range(len(X_np)) if y_np[i] == 1]


# ─────────────────────────────────────────────
# NOUVEAUX SERVEURS À CLASSIFIER
# (cpu %, latence ms)
# ─────────────────────────────────────────────
servers = [
    (42,  78,  "#c77dff", "Serveur A"),   # nominal           → sain
    (85,  75,  "#06d6a0", "Serveur B"),   # CPU élevé         → anomalie
    (40, 200,  "#ffd166", "Serveur C"),   # latence très haute → anomalie
    (95, 210,  "#ff6b6b", "Serveur D"),   # CPU + latence      → anomalie critique
]


# ─────────────────────────────────────────────
# PRÉCISION GLOBALE
# ─────────────────────────────────────────────
with torch.no_grad():
    preds   = model(X).squeeze()
    correct = ((preds > 0.5).float() == y.squeeze()).sum().item()
accuracy = correct / len(y_np)


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Détection d'anomalies réseau — CPU vs latence",
             color="white", fontsize=13)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)


# ── Gauche : frontière de décision ───────────────────────────────────
ax1 = fig.add_subplot(gs[0])
_style(ax1, "Zone de normalité apprise\nbleu = sain  /  rouge = anomalie")

ax1.imshow(ZZ, origin="lower", aspect="auto",
           extent=[cpu_grid[0], cpu_grid[-1], lat_grid[0], lat_grid[-1]],
           cmap="RdBu_r", vmin=0, vmax=1, alpha=0.55)

ax1.scatter(cpu_0, lat_0, color="#7ec8e3", s=22, alpha=0.8, zorder=3, label="sain (classe 0)")
ax1.scatter(cpu_1, lat_1, color="#ff6b6b", s=22, alpha=0.8, zorder=3, label="anomalie (classe 1)")

ax1.axvline(CPU_CENTER, color="#888888", lw=0.8, linestyle=":", alpha=0.5)
ax1.axhline(LAT_CENTER, color="#888888", lw=0.8, linestyle=":", alpha=0.5)
ax1.plot(CPU_CENTER, LAT_CENTER, "*", color="white", markersize=12, zorder=5,
         label=f"nominal ({CPU_CENTER}%, {LAT_CENTER}ms)")

for cpu, lat, col, lbl in servers:
    x_norm = to_norm(cpu, lat)
    x_t    = torch.tensor([x_norm], dtype=torch.float32)
    with torch.no_grad():
        score = model(x_t).item()
    statut = "anomalie" if score > 0.5 else "sain"
    ax1.plot(cpu, lat, "^", color=col, markersize=11, zorder=6,
             markeredgecolor="white", markeredgewidth=0.8)
    ax1.annotate(f"{lbl}\n{score:.0%} → {statut}",
                 xy=(cpu, lat), xytext=(cpu + 4, lat + 12),
                 color=col, fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

ax1.set_xlabel("CPU (%)", color="#888888", fontsize=8)
ax1.set_ylabel("Latence (ms)", color="#888888", fontsize=8)
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7.5, loc="upper right")


# ── Droite : courbe de loss ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
_style(ax2, "Loss sur l'entraînement")

ax2.plot(range(len(losses)), losses, color="#7ec8e3", lw=2)
ax2.set_xlabel("epoch", color="#888888", fontsize=8)
ax2.set_ylabel("BCE loss", color="#888888", fontsize=8)

ax2.text(len(losses) * 0.38, losses[0] * 0.78,
         f"précision : {accuracy:.1%}\nloss finale : {losses[-1]:.4f}\n\n"
         f"Le réseau a appris que\nune anomalie = déviation\ndes métriques normales.\n"
         f"Impossible avec une droite.",
         color="white", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", edgecolor="#333355"))


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "example.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")
print(f"Précision : {accuracy:.1%}")
for cpu, lat, _, lbl in servers:
    x_norm = to_norm(cpu, lat)
    x_t    = torch.tensor([x_norm], dtype=torch.float32)
    with torch.no_grad():
        score = model(x_t).item()
    print(f"  {lbl} (CPU {cpu}%, {lat}ms)  →  {score:.1%}  ({'anomalie' if score > 0.5 else 'sain'})")
