"""
Cas concret : reconnaissance de chiffres sur des tickets de caisse.
Un système lit automatiquement les montants — chaque chiffre est classifié séparément.

python example.py  ->  example.png
Nécessite models/model.pth — lance main.py d'abord.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import CNN

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
model = CNN()
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), "../models/model.pth"),
    weights_only=True
))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST(root=os.path.join(os.path.dirname(__file__), "../data/"),
                            train=False, download=False, transform=transform)


# ─────────────────────────────────────────────
# SÉLECTION D'EXEMPLES — un de chaque chiffre
# ─────────────────────────────────────────────
examples = {}   # digit → (image_tensor, true_label)
for img, label in test_data:
    if label not in examples:
        examples[label] = img
    if len(examples) == 10:
        break

digits_ordered = list(range(10))


# ─────────────────────────────────────────────
# PRÉDICTIONS
# ─────────────────────────────────────────────
def classify(img_tensor):
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()
    return pred, conf, probs


results = {d: classify(examples[d]) for d in digits_ordered}

# Simule un montant lu sur un ticket : "42.75"
receipt_digits = [4, 2, 7, 5]
receipt_label  = "42.75 €"


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("Reconnaissance de chiffres — lecture automatique de tickets de caisse",
             color="white", fontsize=13)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35,
                       width_ratios=[2, 1])


# ── Panel 1 : les 10 chiffres classifiés ─────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(AX_BG)
ax1.set_title("Classification des 10 chiffres (0–9)\nun exemple du test set pour chaque",
              color="white", fontsize=10, pad=8)
ax1.axis("off")

inner = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0, 0],
                                          hspace=0.3, wspace=0.15)
for idx, d in enumerate(digits_ordered):
    ax_i   = fig.add_subplot(inner[idx // 5, idx % 5])
    img_np = examples[d].numpy()[0]
    ax_i.imshow(img_np, cmap="gray", interpolation="nearest")
    pred, conf, _ = results[d]
    color  = "#00ff99" if pred == d else "#ff6b6b"
    symbol = "✓" if pred == d else "✗"
    ax_i.set_title(f"{symbol} {pred}  {conf:.0%}", color=color, fontsize=8, pad=3)
    ax_i.set_xlabel(f"vrai : {d}", color="#888888", fontsize=7, labelpad=2)
    ax_i.set_xticks([]); ax_i.set_yticks([])
    for spine in ax_i.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(1.5)


# ── Panel 2 : simulation ticket de caisse ────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(AX_BG)
ax2.set_title(f'Lecture du montant "{receipt_label}" sur un ticket\nchaque chiffre classifié séparément',
              color="white", fontsize=10, pad=8)
ax2.axis("off")

inner2 = gridspec.GridSpecFromSubplotSpec(1, len(receipt_digits),
                                           subplot_spec=gs[1, 0],
                                           wspace=0.3)

predicted_amount = ""
for idx, d in enumerate(receipt_digits):
    ax_i = fig.add_subplot(inner2[0, idx])
    img_np = examples[d].numpy()[0]
    ax_i.imshow(img_np, cmap="gray", interpolation="nearest")
    pred, conf, _ = results[d]
    predicted_amount += str(pred)
    color = "#00ff99" if pred == d else "#ff6b6b"
    ax_i.set_title(f"→ {pred}\n{conf:.0%}", color=color, fontsize=9, pad=3)
    ax_i.set_xticks([]); ax_i.set_yticks([])
    for spine in ax_i.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(1.5)

# Résultat final
formatted = f"{predicted_amount[:2]}.{predicted_amount[2:]} €"
ax2.text(0.5, -0.18,
         f"Montant lu : {formatted}",
         transform=ax2.transAxes, ha="center", color="#00ff99",
         fontsize=13, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#001a0d", edgecolor="#00ff99", lw=1.5))


# ── Panel 3 : barres de confiance pour un chiffre ────────────────────
ax3 = fig.add_subplot(gs[0, 1])
_style(ax3, "Confiance du modèle sur le chiffre « 7 »\nscore softmax pour chaque classe")

_, _, probs7 = results[7]
colors = ["#00ff99" if i == 7 else "#1d3557" for i in range(10)]
bars   = ax3.barh(range(10), probs7.numpy(), color=colors, edgecolor="#333355")
ax3.set_yticks(range(10))
ax3.set_yticklabels([str(i) for i in range(10)], color="#888888", fontsize=9)
ax3.set_xlabel("score softmax", color="#888888", fontsize=8)
ax3.set_xlim(0, 1)
ax3.invert_yaxis()
ax3.grid(axis="x", color=GRID, linewidth=0.5)
ax3.set_facecolor(AX_BG)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRID)

for i, (bar, p) in enumerate(zip(bars, probs7.numpy())):
    if p > 0.01:
        ax3.text(p + 0.01, i, f"{p:.1%}", va="center",
                 color="white", fontsize=8)


# ── Panel 4 : pourquoi CNN > régression linéaire ─────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(AX_BG)
ax4.axis("off")
ax4.set_title("Ce que le CNN apporte vs une approche naïve",
              color="white", fontsize=10, pad=8)

comparaison = [
    ("Régression linéaire",  "~92%",  "#ff6b6b"),
    ("NN classique",         "~97%",  "#ffd166"),
    ("CNN (ce modèle)",      "~99%",  "#00ff99"),
]
for i, (name, acc, col) in enumerate(comparaison):
    y_ = 0.72 - i * 0.28
    ax4.text(0.05, y_, name, transform=ax4.transAxes,
             color=col, fontsize=10, fontweight="bold", va="center")
    ax4.text(0.72, y_, acc, transform=ax4.transAxes,
             color=col, fontsize=13, fontweight="bold", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a0a",
                       edgecolor=col, lw=1.5))

ax4.text(0.5, 0.08,
         "Le CNN exploite la structure spatiale :\nun filtre 3×3 détecte les mêmes\nbords peu importe où ils se trouvent.",
         transform=ax4.transAxes, ha="center", color="#aaaaaa",
         fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#111122",
                   edgecolor="#333355", lw=1))


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "example.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")

print(f"\nRésultats :")
for d in digits_ordered:
    pred, conf, _ = results[d]
    status = "✓" if pred == d else "✗"
    print(f"  {status} chiffre {d}  →  prédit {pred}  ({conf:.1%})")
print(f"\nMontant lu : {formatted}")
