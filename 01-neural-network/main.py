import torch
import torch.nn as nn
from sklearn.datasets import make_circles

# ── Data ──────────────────────────────────────────────────────────────
X_np, y_np = make_circles(n_samples=1000, noise=0.1, factor=0.4, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)  # (1000,) → (1000, 1)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {y.unique()}")


# ── Model ─────────────────────────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.layer1  = nn.Linear(2, n_hidden)
        self.layer2  = nn.Linear(n_hidden, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


model = NeuralNet(n_hidden=16)
y_hat = model(X)
print(f"y_hat shape: {y_hat.shape}")                            # (1000, 1)
print(f"y_hat range: {y_hat.min():.3f} - {y_hat.max():.3f}")    # entre 0 et 1


# ── Loss & Optimizer ──────────────────────────────────────────────────
loss_fn   = nn.BCELoss()

# Adam = gradient descent amélioré (adapte le lr par paramètre)
# lr=0.01 : à chaque step, les poids bougent de ~1% dans la bonne direction
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ── Training loop ─────────────────────────────────────────────────────
EPOCHS = 500

for epoch in range(EPOCHS):
    # 1. Prédiction (forward pass)
    y_hat = model(X)

    # 2. Loss — mesure l'erreur courante
    loss = loss_fn(y_hat, y)

    # 3. Gradients — PyTorch remonte la chain rule automatiquement
    optimizer.zero_grad()   # remet les gradients à 0 (ils s'accumulent sinon)
    loss.backward()         # calcule dL/dW pour chaque paramètre

    # 4. Update — déplace les poids dans la direction opposée au gradient
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} / {EPOCHS}  —  loss: {loss.item():.4f}")
