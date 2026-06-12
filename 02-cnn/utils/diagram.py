"""
python diagram.py  ->  diagram.png
Diagramme technique : architecture CNN + cycle d'entraînement.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG = "#0f0f1a"

fig, (ax_arch, ax_cycle) = plt.subplots(1, 2, figsize=(20, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("CNN (PyTorch) — architecture & cycle d'entraînement",
             color="white", fontsize=14)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=9):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                   boxstyle="round,pad=0.08",
                                   facecolor=color, edgecolor="white", lw=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.2 if sublabel else 0), label,
            ha="center", va="center", color="white",
            fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.32, sublabel, ha="center", va="center",
                color="#cccccc", fontsize=7.5, zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#7ec8e3", label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.22, label, color=color, fontsize=7.5,
                ha="center", zorder=5)

def code(ax, x, y, txt, color="#aaaaff"):
    ax.text(x, y, txt, color=color, fontsize=8, ha="center", va="center",
            fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#333366", linewidth=1))

def feature_map(ax, x, y, w, h, depth, color, label, sublabel):
    # Représentation 3D d'une feature map en 3 faces
    offx, offy = 0.12, 0.12
    # Face avant
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                   boxstyle="square,pad=0",
                                   facecolor=color, edgecolor="white", lw=1.0, zorder=3,
                                   alpha=0.9)
    ax.add_patch(rect)
    # Face dessus
    xs = [x - w/2, x - w/2 + offx*depth, x + w/2 + offx*depth, x + w/2, x - w/2]
    ys = [y + h/2, y + h/2 + offy*depth, y + h/2 + offy*depth, y + h/2, y + h/2]
    ax.fill(xs, ys, color=color, alpha=0.6, zorder=3)
    ax.plot(xs, ys, color="white", lw=0.8, zorder=4)
    # Face droite
    xs2 = [x + w/2, x + w/2 + offx*depth, x + w/2 + offx*depth, x + w/2, x + w/2]
    ys2 = [y - h/2, y - h/2 + offy*depth, y + h/2 + offy*depth, y + h/2, y - h/2]
    ax.fill(xs2, ys2, color=color, alpha=0.45, zorder=3)
    ax.plot(xs2, ys2, color="white", lw=0.8, zorder=4)
    # Labels
    ax.text(x, y + 0.18, label, ha="center", va="center",
            color="white", fontsize=8, fontweight="bold", zorder=5)
    ax.text(x, y - 0.25, sublabel, ha="center", va="center",
            color="#cccccc", fontsize=7, zorder=5)


# ─────────────────────────────────────────────
# GAUCHE : Architecture CNN
# ─────────────────────────────────────────────
ax = ax_arch
ax.set_facecolor(BG)
ax.set_xlim(-0.5, 13)
ax.set_ylim(-1, 10)
ax.axis("off")
ax.set_title("Architecture CNN", color="white", fontsize=12, pad=12)

Y = 5.0

# Input
feature_map(ax, 0.5, Y, 0.9, 1.4, 1, "#1d3557", "28×28", "1ch")
ax.text(0.5, Y + 1.4, "INPUT", ha="center", color="#7ec8e3", fontsize=8)

arrow(ax, 1.1, Y, 1.6, Y)

# Conv1
feature_map(ax, 2.2, Y, 0.9, 1.2, 4, "#2d6a4f", "26×26", "32ch")
ax.text(2.2, Y + 1.2, "Conv2d(1,32,3)\n+ ReLU", ha="center", color="#7ec8e3", fontsize=7.5)

arrow(ax, 2.8, Y, 3.3, Y)

# Pool1
feature_map(ax, 3.9, Y, 0.75, 0.95, 4, "#2d6a4f", "13×13", "32ch")
ax.text(3.9, Y + 1.0, "MaxPool2d(2)", ha="center", color="#7ec8e3", fontsize=7.5)

arrow(ax, 4.4, Y, 4.9, Y)

# Conv2
feature_map(ax, 5.6, Y, 0.7, 0.88, 5, "#7b2d00", "11×11", "64ch")
ax.text(5.6, Y + 1.0, "Conv2d(32,64,3)\n+ ReLU", ha="center", color="#7ec8e3", fontsize=7.5)

arrow(ax, 6.1, Y, 6.6, Y)

# Pool2
feature_map(ax, 7.2, Y, 0.55, 0.68, 5, "#7b2d00", "5×5", "64ch")
ax.text(7.2, Y + 0.85, "MaxPool2d(2)", ha="center", color="#7ec8e3", fontsize=7.5)

arrow(ax, 7.65, Y, 8.1, Y)

# Flatten
box(ax, 8.7, Y, 0.9, 0.7, "#4a1942", "Flatten", "1600")
ax.text(8.7, Y + 0.65, "Flatten()", ha="center", color="#7ec8e3", fontsize=7.5)

arrow(ax, 9.2, Y, 9.7, Y)

# FC
box(ax, 10.3, Y, 0.9, 0.7, "#1a3a4a", "Linear", "1600→10")
ax.text(10.3, Y + 0.65, "Linear(1600,10)", ha="center", color="#7ec8e3", fontsize=7.5)

arrow(ax, 10.8, Y, 11.3, Y)

# Output
for i, (digit, dy) in enumerate([(0,2.2),(1,1.4),(2,0.6),(3,-0.2),
                                   (4,-1.0),(5,-1.8),(6,-2.6),(7,-3.4),
                                   (8,-4.2),(9,-5.0)]):
    col = "#00ff99" if digit == 7 else "#1d3557"
    rect = mpatches.FancyBboxPatch((11.35, Y + dy - 0.28), 0.7, 0.48,
                                   boxstyle="round,pad=0.04",
                                   facecolor=col, edgecolor="white", lw=0.8, zorder=3)
    ax.add_patch(rect)
    ax.text(11.7, Y + dy, str(digit), ha="center", va="center",
            color="white", fontsize=8, fontweight="bold", zorder=4)

ax.text(11.7, Y + 2.9, "OUTPUT\n(10 scores)", ha="center", color="#7ec8e3", fontsize=8)

# Note kernel
ax.text(2.2, Y - 2.2,
        "filtre 3×3 : regarde 9 pixels voisins à la fois\nglisse sur toute l'image → détecte bords, courbes",
        color="#ffd166", fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1500", edgecolor="#ffd166", lw=1))

ax.text(5.6, Y - 2.2,
        "32 cartes → 64 cartes\npatterns plus complexes\n(boucles, jonctions)",
        color="#ff6b6b", fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a0000", edgecolor="#ff6b6b", lw=1))


# ─────────────────────────────────────────────
# DROITE : Cycle d'entraînement
# ─────────────────────────────────────────────
ax2 = ax_cycle
ax2.set_facecolor(BG)
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.5, 10.5)
ax2.axis("off")
ax2.set_title("Cycle d'entraînement PyTorch", color="white", fontsize=12, pad=12)

box(ax2, 1.8, 8.8, 2.8, 1.1, "#1d3557",  "1. DATA",       "DataLoader (batch=64)")
box(ax2, 5.2, 8.8, 2.8, 1.1, "#2d6a4f",  "2. FORWARD",    "model(X_batch)")
box(ax2, 8.2, 6.8, 2.8, 1.1, "#7b2d00",  "3. LOSS",       "CrossEntropyLoss")
box(ax2, 8.2, 4.2, 2.8, 1.1, "#4a1942",  "4. BACKWARD",   "autograd")
box(ax2, 5.2, 4.2, 2.8, 1.1, "#1a3a4a",  "5. UPDATE",     "optimizer.step()")
box(ax2, 1.8, 4.2, 2.8, 1.1, "#2a2a00",  "6. LOOP",       "937 batches × 5 epochs")

code(ax2, 5.2, 7.5,  "y_hat = model(X_batch)   # (64, 10)")
code(ax2, 8.2, 5.6,  "loss = loss_fn(y_hat, y_batch)")
code(ax2, 8.2, 3.0,  "optimizer.zero_grad()\nloss.backward()")
code(ax2, 5.2, 3.0,  "optimizer.step()")

arrow(ax2, 3.2, 8.8, 3.8, 8.8)
arrow(ax2, 6.6, 8.8, 6.8, 7.3)
arrow(ax2, 8.2, 6.3, 8.2, 4.8)
arrow(ax2, 6.8, 4.2, 6.6, 4.2)
arrow(ax2, 3.8, 4.2, 3.2, 4.2)

ax2.annotate("", xy=(5.2, 8.25), xytext=(1.8, 4.76),
             arrowprops=dict(arrowstyle="-|>", color="#ffd166",
                             lw=2.0, mutation_scale=14,
                             connectionstyle="arc3,rad=0.35"), zorder=2)
ax2.text(2.0, 6.6, "boucle\nd'entraînement",
         color="#ffd166", fontsize=8, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1500", edgecolor="#ffd166"))

# Différence clé vs NN
ax2.text(5.0, 1.4,
         "Différence vs NN :  DataLoader passe 937 mini-batches de 64 images par epoch\n"
         "au lieu d'un seul forward sur tout le dataset — même mécanique, plus scalable",
         color="#00ff99", fontsize=8.5, ha="center",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#001a0d", edgecolor="#00ff99", lw=1))

ax2.text(5.0, 0.3,
         "CrossEntropyLoss  =  softmax + log  —  généralise BCELoss à N classes",
         color="#aaaaff", fontsize=8, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a", edgecolor="#333366", lw=1))


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "diagram.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")
