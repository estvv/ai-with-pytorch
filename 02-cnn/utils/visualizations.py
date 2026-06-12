"""
python visualizations.py  ->  training.png
Ce que le CNN a appris : filtres conv1, prédictions, courbes loss/accuracy.
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

BG, AX_BG, GRID_COL = "#0f0f1a", "#13132b", "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5)


# ─────────────────────────────────────────────
# CHARGEMENT MODÈLE + DATA
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
test_data   = datasets.MNIST(root=os.path.join(os.path.dirname(__file__), "../data/"),
                              train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)


# ─────────────────────────────────────────────
# RÉENTRAÎNEMENT POUR CAPTURER L'HISTORIQUE
# ─────────────────────────────────────────────
train_data   = datasets.MNIST(root=os.path.join(os.path.dirname(__file__), "../data/"),
                               train=True, download=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model_hist   = CNN()
loss_fn      = torch.nn.CrossEntropyLoss()
optimizer    = torch.optim.Adam(model_hist.parameters(), lr=0.001)
EPOCHS       = 5
hist_loss, hist_acc = [], []

print("Entraînement pour capturer l'historique...")
for epoch in range(EPOCHS):
    model_hist.train()
    total_loss = 0
    for X_b, y_b in train_loader:
        out  = model_hist(X_b)
        loss = loss_fn(out, y_b)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    hist_loss.append(total_loss / len(train_loader))

    model_hist.eval()
    correct = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            correct += (model_hist(X_b).argmax(1) == y_b).sum().item()
    hist_acc.append(correct / len(test_data))
    print(f"  Epoch {epoch+1}/{EPOCHS}  loss={hist_loss[-1]:.4f}  acc={hist_acc[-1]:.2%}")


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor(BG)
fig.suptitle("CNN (PyTorch) — ce que le réseau a appris",
             color="white", fontsize=14, y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)


# ── Panel 1 : filtres conv1 (les 32 filtres 3×3) ─────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(AX_BG)
ax1.set_title("Filtres conv1 — ce que chaque filtre détecte\n(bords, orientations, textures)",
              color="white", fontsize=10, pad=8)
ax1.axis("off")

weights = model.conv1.weight.data.cpu().numpy()  # (32, 1, 3, 3)
weights = weights[:, 0, :, :]                     # (32, 3, 3)

grid_size = 8   # 8×4 = 32 filtres
inner = gridspec.GridSpecFromSubplotSpec(4, grid_size, subplot_spec=gs[0, 0],
                                         hspace=0.05, wspace=0.05)
for idx in range(32):
    ax_f = fig.add_subplot(inner[idx // grid_size, idx % grid_size])
    ax_f.imshow(weights[idx], cmap="RdBu_r", interpolation="nearest")
    ax_f.axis("off")


# ── Panel 2 : prédictions correctes ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(AX_BG)
ax2.set_title("Prédictions correctes\n(confiance > 99%)",
              color="white", fontsize=10, pad=8)
ax2.axis("off")

correct_imgs, correct_labels = [], []
with torch.no_grad():
    for X_b, y_b in DataLoader(test_data, batch_size=512, shuffle=True):
        probs  = torch.softmax(model(X_b), dim=1)
        preds  = probs.argmax(1)
        confs  = probs.max(1).values
        mask   = (preds == y_b) & (confs > 0.99)
        correct_imgs.extend(X_b[mask][:20].tolist())
        correct_labels.extend(list(zip(preds[mask][:20].tolist(),
                                       confs[mask][:20].tolist())))
        if len(correct_imgs) >= 20:
            break

inner2 = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs[0, 1],
                                           hspace=0.1, wspace=0.1)
for idx in range(min(20, len(correct_imgs))):
    ax_i = fig.add_subplot(inner2[idx // 5, idx % 5])
    img  = np.array(correct_imgs[idx][0])
    ax_i.imshow(img, cmap="gray", interpolation="nearest")
    digit, conf = correct_labels[idx]
    ax_i.set_title(f"{digit} {conf:.0%}", color="#00ff99", fontsize=7, pad=2)
    ax_i.axis("off")


# ── Panel 3 : prédictions incorrectes ────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(AX_BG)
ax3.set_title("Erreurs du modèle\n(vrai → prédit)",
              color="white", fontsize=10, pad=8)
ax3.axis("off")

wrong_imgs, wrong_labels = [], []
with torch.no_grad():
    for X_b, y_b in DataLoader(test_data, batch_size=512, shuffle=True):
        preds = model(X_b).argmax(1)
        mask  = preds != y_b
        wrong_imgs.extend(X_b[mask][:20].tolist())
        wrong_labels.extend(list(zip(y_b[mask][:20].tolist(),
                                     preds[mask][:20].tolist())))
        if len(wrong_imgs) >= 20:
            break

inner3 = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs[0, 2],
                                           hspace=0.1, wspace=0.1)
for idx in range(min(20, len(wrong_imgs))):
    ax_i = fig.add_subplot(inner3[idx // 5, idx % 5])
    img  = np.array(wrong_imgs[idx][0])
    ax_i.imshow(img, cmap="gray", interpolation="nearest")
    true, pred = wrong_labels[idx]
    ax_i.set_title(f"{true}→{pred}", color="#ff6b6b", fontsize=7, pad=2)
    ax_i.axis("off")


# ── Panel 4 : courbe de loss ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
_style(ax4, "Loss par epoch")
ax4.plot(range(1, EPOCHS+1), hist_loss, color="#7ec8e3", lw=2, marker="o", markersize=6)
ax4.set_xlabel("epoch", color="#888888", fontsize=8)
ax4.set_ylabel("CrossEntropy loss", color="#888888", fontsize=8)
ax4.set_xticks(range(1, EPOCHS+1))
ax4.text(EPOCHS * 0.6, hist_loss[0] * 0.85,
         f"loss finale : {hist_loss[-1]:.4f}",
         color="white", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333355"))


# ── Panel 5 : courbe accuracy ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
_style(ax5, "Accuracy sur le test set")
ax5.plot(range(1, EPOCHS+1), [a * 100 for a in hist_acc],
         color="#00ff99", lw=2, marker="o", markersize=6)
ax5.set_xlabel("epoch", color="#888888", fontsize=8)
ax5.set_ylabel("accuracy (%)", color="#888888", fontsize=8)
ax5.set_xticks(range(1, EPOCHS+1))
ax5.set_ylim(90, 100)
ax5.text(EPOCHS * 0.55, hist_acc[0] * 100 - 2,
         f"accuracy finale : {hist_acc[-1]:.2%}",
         color="white", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333355"))


# ── Panel 6 : matrice de confusion ───────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
_style(ax6, "Matrice de confusion\n(où le modèle se trompe)")

conf_matrix = np.zeros((10, 10), dtype=int)
with torch.no_grad():
    for X_b, y_b in test_loader:
        preds = model(X_b).argmax(1)
        for t, p in zip(y_b.tolist(), preds.tolist()):
            conf_matrix[t][p] += 1

im = ax6.imshow(conf_matrix, cmap="Blues", aspect="auto")
ax6.set_xticks(range(10)); ax6.set_yticks(range(10))
ax6.set_xticklabels(range(10), color="#888888", fontsize=8)
ax6.set_yticklabels(range(10), color="#888888", fontsize=8)
ax6.set_xlabel("prédit", color="#888888", fontsize=8)
ax6.set_ylabel("vrai", color="#888888", fontsize=8)
ax6.grid(False)

for i in range(10):
    for j in range(10):
        val = conf_matrix[i][j]
        if val > 0:
            color = "white" if conf_matrix[i][j] > conf_matrix.max() * 0.5 else "#888888"
            ax6.text(j, i, str(val), ha="center", va="center",
                     fontsize=6.5, color=color)


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "training.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")
