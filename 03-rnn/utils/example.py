"""
Cas concret : assistant de rédaction stylistique.
Le RNN continue n'importe quelle phrase dans le style des Misérables.

python example.py  ->  example.png
Nécessite models/model.pth — lance main.py d'abord.
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import RNN

BG, AX_BG, GRID = "#0f0f1a", "#13132b", "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.6)


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
ckpt = torch.load(os.path.join(os.path.dirname(__file__), "../models/model.pth"),
                  weights_only=False)
chars = ckpt["vocab"]
c2i   = {c: i for i, c in enumerate(chars)}
i2c   = {i: c for i, c in enumerate(chars)}

model = RNN(len(chars), ckpt["embed_dim"], ckpt["hidden_dim"], ckpt["n_layers"])
model.load_state_dict(ckpt["model_state"])
model.eval()


# ─────────────────────────────────────────────
# GÉNÉRATION
# ─────────────────────────────────────────────
def generate(seed, length=300, temperature=0.7):
    known = [c for c in seed if c in c2i]
    if not known:
        return seed
    x = torch.tensor([c2i[c] for c in known], dtype=torch.long).unsqueeze(0)
    result = ""
    h = None
    with torch.no_grad():
        out, h = model(x, h)
        for _ in range(length):
            logits    = out[0, -1, :] / temperature
            probs     = torch.softmax(logits, dim=0)
            next_char = torch.multinomial(probs, 1).item()
            result   += i2c[next_char]
            x         = torch.tensor([[next_char]], dtype=torch.long)
            out, h    = model(x, h)
    return result


# Scénarios : seed connue du corpus + seed inventée
scenarios = [
    ("Extrait du corpus",
     "Jean Valjean ",
     "#7ec8e3", 0.6),
    ("Personnage inventé",
     "Marie regarda ",
     "#ffd166", 0.7),
    ("Lieu inventé",
     "La ville de Lyon ",
     "#ff6b6b", 0.7),
    ("Début de dialogue",
     "— Que voulez-vous ",
     "#c77dff", 0.75),
]


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("Assistant stylistique — continuation dans le style des Misérables (Victor Hugo)",
             color="white", fontsize=13)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)


for idx, (title, seed, color, temp) in enumerate(scenarios):
    row, col = idx // 2, idx % 2
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor(AX_BG)
    ax.set_title(f"{title}  —  seed : \"{seed}\"",
                 color=color, fontsize=10, pad=8)
    ax.axis("off")

    completion = generate(seed, length=280, temperature=temp)
    full_text  = seed + completion

    # Découpe en lignes de ~60 chars
    lines = []
    while full_text:
        cut = min(62, len(full_text))
        # coupe proprement sur un espace si possible
        if cut < len(full_text) and ' ' in full_text[:cut]:
            cut = full_text[:cut].rfind(' ') + 1
        lines.append(full_text[:cut])
        full_text = full_text[cut:]
        if len(lines) >= 12:
            break

    for l_idx, line in enumerate(lines):
        y_ = 0.90 - l_idx * 0.077
        ax.text(0.02, y_, line, transform=ax.transAxes,
                color="#dddddd", fontsize=8.5, fontfamily="monospace", va="top")

    # Badge température
    ax.text(0.97, 0.04, f"t={temp}", transform=ax.transAxes,
            color=color, fontsize=8, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a1a",
                      edgecolor=color, lw=1))


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "example.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out_path}")
