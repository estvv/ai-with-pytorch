"""
python visualizations.py  ->  training.png
Ce que le RNN a appris : loss, texte généré à différentes températures,
évolution de la qualité par epoch.
Nécessite models/model.pth — lance main.py d'abord.
"""
import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import RNN

BG, AX_BG, GRID_COL = "#0f0f1a", "#13132b", "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5)


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
ckpt = torch.load(os.path.join(os.path.dirname(__file__), "../models/model.pth"),
                  weights_only=False)
chars = ckpt["vocab"]
c2i   = {c: i for i, c in enumerate(chars)}
i2c   = {i: c for i, c in enumerate(chars)}
VOCAB = len(chars)

model = RNN(VOCAB, ckpt["embed_dim"], ckpt["hidden_dim"], ckpt["n_layers"])
model.load_state_dict(ckpt["model_state"])
model.eval()


# ─────────────────────────────────────────────
# RÉENTRAÎNEMENT AVEC HISTORIQUE
# ─────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "../data/miserables_fr.txt"),
          "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

data = torch.tensor([c2i[c] for c in text if c in c2i], dtype=torch.long)

SEQ_LEN, BATCH_SIZE = 100, 128

def make_batches(data, seq_len, batch_size):
    n_seq = (len(data) - 1) // seq_len
    n_seq = (n_seq // batch_size) * batch_size
    inputs  = torch.stack([data[i*seq_len : i*seq_len+seq_len]   for i in range(n_seq)])
    targets = torch.stack([data[i*seq_len+1 : i*seq_len+seq_len+1] for i in range(n_seq)])
    return inputs.view(-1, batch_size, seq_len), targets.view(-1, batch_size, seq_len)

inputs, targets = make_batches(data, SEQ_LEN, BATCH_SIZE)

model_hist = RNN(VOCAB, ckpt["embed_dim"], ckpt["hidden_dim"], ckpt["n_layers"])
loss_fn    = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model_hist.parameters(), lr=0.002)
EPOCHS     = 10
hist_loss  = []

print("Entraînement pour capturer l'historique...")
for epoch in range(EPOCHS):
    model_hist.train()
    total_loss = 0
    h = None
    for i in range(inputs.shape[0]):
        if h is not None:
            h = tuple(s.detach() for s in h)
        out, h = model_hist(inputs[i], h)
        loss = loss_fn(out.permute(0, 2, 1), targets[i])
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model_hist.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    hist_loss.append(total_loss / inputs.shape[0])
    print(f"  Epoch {epoch+1}/{EPOCHS}  loss={hist_loss[-1]:.4f}")


# ─────────────────────────────────────────────
# GÉNÉRATION
# ─────────────────────────────────────────────
def generate(seed, length=250, temperature=0.7):
    known = [c for c in seed if c in c2i]
    if not known:
        return ""
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


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor(BG)
fig.suptitle("RNN (PyTorch) — entraîné sur Les Misérables",
             color="white", fontsize=14, y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)


# ── Panel 1 : courbe de loss ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
_style(ax1, "Loss par epoch")
ax1.plot(range(1, EPOCHS+1), hist_loss, color="#7ec8e3", lw=2, marker="o", markersize=6)
ax1.set_xlabel("epoch", color="#888888", fontsize=8)
ax1.set_ylabel("CrossEntropy loss", color="#888888", fontsize=8)
ax1.set_xticks(range(1, EPOCHS+1))
ax1.text(EPOCHS * 0.55, hist_loss[0] * 0.88,
         f"loss finale : {hist_loss[-1]:.4f}",
         color="white", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333355"))


# ── Panel 2 : effet de la température ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(AX_BG)
ax2.set_title("Effet de la température sur la génération\n(même seed : \"Jean Valjean \")",
              color="white", fontsize=10, pad=8)
ax2.axis("off")

seed = "Jean Valjean "
temps = [(0.3, "#7ec8e3", "0.3 — conservateur"), (0.7, "#ffd166", "0.7 — équilibré"),
         (1.2, "#ff6b6b", "1.2 — créatif")]
y_pos = 0.92
for temp, col, label in temps:
    text_gen = generate(seed, length=120, temperature=temp)
    ax2.text(0.02, y_pos, label, transform=ax2.transAxes,
             color=col, fontsize=8.5, fontweight="bold")
    y_pos -= 0.06
    # Wrap le texte sur ~50 chars
    words = text_gen
    lines = [words[i:i+55] for i in range(0, min(len(words), 165), 55)]
    for line in lines[:3]:
        ax2.text(0.02, y_pos, line, transform=ax2.transAxes,
                 color="#cccccc", fontsize=7.5, fontfamily="monospace")
        y_pos -= 0.055
    y_pos -= 0.04


# ── Panel 3 : exemples de complétion ─────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(AX_BG)
ax3.set_title("Complétion depuis différents seeds",
              color="white", fontsize=10, pad=8)
ax3.axis("off")

seeds = ["Il dit", "La nuit", "Paris ", "Cosette "]
y_pos = 0.92
for s in seeds:
    gen = generate(s, length=100, temperature=0.7)
    ax3.text(0.02, y_pos, f"→ \"{s}\"", transform=ax3.transAxes,
             color="#00ff99", fontsize=8.5, fontweight="bold")
    y_pos -= 0.06
    lines = (s + gen)
    display = lines[:110]
    chunks = [display[i:i+50] for i in range(0, len(display), 50)]
    for chunk in chunks[:2]:
        ax3.text(0.02, y_pos, chunk, transform=ax3.transAxes,
                 color="#cccccc", fontsize=7.5, fontfamily="monospace")
        y_pos -= 0.055
    y_pos -= 0.04


# ── Panel 4 : distribution des caractères générés ────────────────────
ax4 = fig.add_subplot(gs[1, 0])
_style(ax4, "Distribution — corpus vs généré\n(top 20 caractères)")

long_gen = generate("Il était une fois ", length=2000, temperature=0.7)
import collections
corpus_sample = text[:10000]
corp_freq = collections.Counter(corpus_sample)
gen_freq  = collections.Counter(long_gen)

top_chars = [c for c, _ in corp_freq.most_common(20)]
corp_vals = [corp_freq[c] / len(corpus_sample) for c in top_chars]
gen_vals  = [gen_freq.get(c, 0) / len(long_gen) for c in top_chars]

x = range(len(top_chars))
ax4.bar([i - 0.2 for i in x], corp_vals, 0.38, label="corpus", color="#1d3557", edgecolor="#333355")
ax4.bar([i + 0.2 for i in x], gen_vals,  0.38, label="généré",  color="#7ec8e3", edgecolor="#333355", alpha=0.85)
ax4.set_xticks(list(x))
ax4.set_xticklabels([repr(c)[1:-1] for c in top_chars], color="#888888", fontsize=7, rotation=45)
ax4.set_ylabel("fréquence", color="#888888", fontsize=8)
ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)


# ── Panel 5 : perplexité par epoch ───────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
_style(ax5, "Perplexité par epoch\n(exp(loss) — plus bas = mieux)")
import math
perplexities = [math.exp(l) for l in hist_loss]
ax5.plot(range(1, EPOCHS+1), perplexities, color="#ff6b6b", lw=2, marker="o", markersize=6)
ax5.set_xlabel("epoch", color="#888888", fontsize=8)
ax5.set_ylabel("perplexité", color="#888888", fontsize=8)
ax5.set_xticks(range(1, EPOCHS+1))
ax5.text(0.55, 0.85,
         f"perplexité finale :\n{perplexities[-1]:.1f} caractères\npossibles en moyenne",
         transform=ax5.transAxes, color="white", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333355"))


# ── Panel 6 : RNN vs CNN vs NN ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(AX_BG)
ax6.set_title("Comparaison des architectures", color="white", fontsize=10, pad=8)
ax6.axis("off")

rows = [
    ("",         "Mémoire",    "Entrée",         "Tâche typique"),
    ("NN",       "Aucune",     "vecteur fixe",   "classification"),
    ("CNN",      "Locale",     "image 2D",       "vision"),
    ("RNN/LSTM", "Séquentielle","séquence",      "texte, séries"),
    ("Transformer","Globale",  "séquence",       "LLM, GPT…"),
]
col_x = [0.02, 0.22, 0.45, 0.68]
col_colors = ["#7ec8e3", "#aaaaaa", "#aaaaaa", "#aaaaaa"]

for r_idx, row in enumerate(rows):
    y_ = 0.90 - r_idx * 0.17
    is_header = r_idx == 0
    is_current = r_idx == 3
    for c_idx, (val, cx) in enumerate(zip(row, col_x)):
        color = "#7ec8e3" if is_header else ("#00ff99" if is_current else "#cccccc")
        fw = "bold" if (is_header or is_current) else "normal"
        ax6.text(cx, y_, val, transform=ax6.transAxes,
                 color=color, fontsize=8.5 if is_current else 8,
                 fontweight=fw, va="center")

ax6.axhline(y=0.83, xmin=0.01, xmax=0.99, color="#333355", lw=0.8)


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "training.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out_path}")
