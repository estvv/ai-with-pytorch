import os
import torch
import torch.nn as nn
from model import RNN

# ── Data ──────────────────────────────────────────────────────────────
with open("data/miserables_fr.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

chars  = sorted(set(text))
VOCAB  = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}
data = torch.tensor([c2i[c] for c in text], dtype=torch.long)

print(f"Corpus     : {len(text):,} caractères")
print(f"Vocabulaire: {VOCAB} caractères uniques")
print(f"Exemple    : {text[:50]!r}")
print(f"Data shape : {data.shape}")


# ── Séquences ─────────────────────────────────────────────────────────
SEQ_LEN    = 100
BATCH_SIZE = 128

def make_batches(data, seq_len, batch_size):
    n_seq = (len(data) - 1) // seq_len
    n_seq = (n_seq // batch_size) * batch_size
    inputs  = torch.stack([data[i*seq_len : i*seq_len + seq_len]   for i in range(n_seq)])
    targets = torch.stack([data[i*seq_len+1 : i*seq_len + seq_len+1] for i in range(n_seq)])
    inputs  = inputs.view(-1, batch_size, seq_len)
    targets = targets.view(-1, batch_size, seq_len)
    return inputs, targets

inputs, targets = make_batches(data, SEQ_LEN, BATCH_SIZE)

print(f"Batches    : {inputs.shape[0]}")
print(f"inputs[0]  : {inputs.shape}")
print(f"Exemple input  : {''.join(i2c[i.item()] for i in inputs[0][0][:30])!r}")
print(f"Exemple target : {''.join(i2c[i.item()] for i in targets[0][0][:30])!r}")


# ── Model ─────────────────────────────────────────────────────────────
EMBED_DIM  = 64
HIDDEN_DIM = 256
N_LAYERS   = 2

model = RNN(VOCAB, EMBED_DIM, HIDDEN_DIM, N_LAYERS)

x_test     = inputs[0][:4]
out, state = model(x_test)
print(f"output shape : {out.shape}")
print(f"h shape      : {state[0].shape}")


# ── Loss & Optimizer ──────────────────────────────────────────────────
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


# ── Training loop ─────────────────────────────────────────────────────
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    h = None

    for i in range(inputs.shape[0]):
        x_batch = inputs[i]
        y_batch = targets[i]

        if h is not None:
            h = tuple(s.detach() for s in h)

        out, h = model(x_batch, h)
        loss = loss_fn(out.permute(0, 2, 1), y_batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / inputs.shape[0]
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  —  loss: {avg_loss:.4f}")


# ── Sauvegarde ────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "vocab":       chars,
    "embed_dim":   EMBED_DIM,
    "hidden_dim":  HIDDEN_DIM,
    "n_layers":    N_LAYERS,
}, "models/model.pth")
print("Modèle sauvegardé -> models/model.pth")
