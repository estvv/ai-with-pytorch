"""
python diagram.py  ->  diagram.png
Diagramme technique : architecture RNN/LSTM + cycle d'entraînement.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

BG = "#0f0f1a"

fig, (ax_arch, ax_cycle) = plt.subplots(1, 2, figsize=(20, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("RNN / LSTM (PyTorch) — architecture & cycle d'entraînement",
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
        ax.text(x, y - 0.3, sublabel, ha="center", va="center",
                color="#cccccc", fontsize=7.5, zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#7ec8e3", label=None, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.6, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.2, label, color=color, fontsize=7.5, ha="center", zorder=5)

def code(ax, x, y, txt, color="#aaaaff"):
    ax.text(x, y, txt, color=color, fontsize=8, ha="center", va="center",
            fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#333366", linewidth=1))


# ─────────────────────────────────────────────
# GAUCHE : Architecture LSTM déroulée dans le temps
# ─────────────────────────────────────────────
ax = ax_arch
ax.set_facecolor(BG)
ax.set_xlim(-0.5, 13)
ax.set_ylim(-1.5, 10)
ax.axis("off")
ax.set_title("Architecture LSTM — déroulée dans le temps", color="white", fontsize=12, pad=12)

# Titre étapes
ax.text(6.0, 9.5, "\"Jean \" → \"Jean V\" → \"Jean Va\" → … → \"Jean Valjean \"",
        ha="center", color="#888888", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#111122", edgecolor="#333355", lw=1))

# 4 cellules LSTM déroulées
positions = [1.5, 4.5, 7.5, 10.5]
chars_in  = ['"J"', '"e"', '"a"', '"n"']
chars_out = ['"e"', '"a"', '"n"', '" "']

for i, (px, cin, cout) in enumerate(zip(positions, chars_in, chars_out)):
    # Cellule LSTM
    box(ax, px, 5.0, 2.2, 1.8, "#2d6a4f", "LSTM", f"t={i+1}", fontsize=10)

    # Input
    box(ax, px, 2.5, 1.4, 0.8, "#1d3557", cin)
    arrow(ax, px, 2.9, px, 4.1)

    # Embedding
    ax.text(px, 3.6, "embed", color="#888888", fontsize=7, ha="center")

    # Output → prédit le suivant
    box(ax, px, 7.5, 1.4, 0.8, "#7b2d00", cout)
    arrow(ax, px, 5.9, px, 7.1)
    ax.text(px, 6.7, "fc", color="#888888", fontsize=7, ha="center")

    # Flèche état caché h → cellule suivante
    if i < len(positions) - 1:
        arrow(ax, px + 1.1, 5.0, positions[i+1] - 1.1, 5.0, "#ffd166", "h")

# Légende état caché
ax.annotate("", xy=(10.5 + 1.1, 5.0), xytext=(10.5 + 1.6, 5.0),
            arrowprops=dict(arrowstyle="-|>", color="#ffd166", lw=1.6, mutation_scale=14))
ax.text(11.8, 5.0, "h\n→ ∞", color="#ffd166", fontsize=8, ha="left", va="center")

# Labels colonnes
ax.text(-0.2, 2.5, "INPUT\n(char)", color="#7ec8e3", fontsize=8.5, ha="center")
ax.text(-0.2, 7.5, "OUTPUT\n(next char\nprédit)", color="#7ec8e3", fontsize=8.5, ha="center")
ax.text(-0.2, 5.0, "LSTM\ncell", color="#7ec8e3", fontsize=8.5, ha="center")

# "..." à droite
ax.text(12.2, 5.0, "…", color="#888888", fontsize=18, ha="center", va="center")

# Embedding explanation
ax.text(6.0, 0.8,
        "Embedding : 'J' → entier 42 → vecteur (64,)   —   le réseau apprend une représentation dense de chaque caractère\n"
        "État caché h : résumé de tout ce qui a été lu jusqu'à t   —   se propage à travers toute la séquence",
        color="#aaaaaa", fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a1a", edgecolor="#333355", lw=1))


# ─────────────────────────────────────────────
# DROITE : Cycle d'entraînement
# ─────────────────────────────────────────────
ax2 = ax_cycle
ax2.set_facecolor(BG)
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.5, 10.5)
ax2.axis("off")
ax2.set_title("Cycle d'entraînement PyTorch", color="white", fontsize=12, pad=12)

box(ax2, 1.8, 8.8, 2.8, 1.1, "#1d3557",  "1. DATA",       "corpus → séquences de 100 chars")
box(ax2, 5.2, 8.8, 2.8, 1.1, "#2d6a4f",  "2. FORWARD",    "out, h = model(x, h)")
box(ax2, 8.2, 6.8, 2.8, 1.1, "#7b2d00",  "3. LOSS",       "CrossEntropyLoss")
box(ax2, 8.2, 4.2, 2.8, 1.1, "#4a1942",  "4. BACKWARD",   "loss.backward()")
box(ax2, 5.2, 4.2, 2.8, 1.1, "#1a3a4a",  "5. UPDATE",     "optimizer.step()")
box(ax2, 1.8, 4.2, 2.8, 1.1, "#2a2a00",  "6. LOOP",       "49 batches × 10 epochs")

code(ax2, 5.2, 7.5,  "out, h = model(x_batch, h)")
code(ax2, 8.2, 5.6,  "loss = loss_fn(out.permute(0,2,1), y)")
code(ax2, 8.2, 3.0,  "h = tuple(s.detach() for s in h)\nloss.backward()\nclip_grad_norm_(..., 1.0)")
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

ax2.text(5.0, 1.5,
         "h.detach() : coupe le graphe entre batches — évite de backpropager sur toute l'histoire\n"
         "clip_grad_norm_ : plafonne les gradients à 1.0 — évite les explosions dans les RNN",
         color="#00ff99", fontsize=8.5, ha="center",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#001a0d", edgecolor="#00ff99", lw=1))

ax2.text(5.0, 0.3,
         "Différence clé vs CNN : le modèle reçoit h en entrée ET le retourne — la séquence a une mémoire",
         color="#aaaaff", fontsize=8, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a", edgecolor="#333366", lw=1))


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "diagram.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")
