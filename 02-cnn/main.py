import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

# ── Data ──────────────────────────────────────────────────────────────
# ToTensor() : convertit l'image (0–255) en tenseur float (0.0–1.0)
# Normalize() : centre les valeurs — mean=0.1307, std=0.3081 sont les
#               valeurs exactes calculées sur tout MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root="data/", train=True,  download=False, transform=transform)
test_data  = datasets.MNIST(root="data/", train=False, download=False, transform=transform)

# DataLoader : découpe les données en mini-batches de 64 images
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"Train : {len(train_data)} images")
print(f"Test  : {len(test_data)} images")

# Inspecte un batch
X_batch, y_batch = next(iter(train_loader))
print(f"Batch X shape : {X_batch.shape}")   # (64, 1, 28, 28)
print(f"Batch y shape : {y_batch.shape}")   # (64,)
print(f"Classes       : {y_batch.unique()}")

# ── Model ─────────────────────────────────────────────────────────────
# défini dans model.py — importé en haut

model = CNN()
y_hat = model(X_batch)
print(f"y_hat shape : {y_hat.shape}")   # (64, 10) — un score par classe
print(f"prédiction  : {y_hat[0].argmax().item()}")  # classe prédite pour la 1re image
print(f"vrai label  : {y_batch[0].item()}")

# ── Loss & Optimizer ──────────────────────────────────────────────────
# CrossEntropyLoss = softmax + log + negative likelihood — pour N classes
# équivalent de ta BCELoss mais généralisé à 10 classes
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ── Training loop ─────────────────────────────────────────────────────
EPOCHS = 5  # 5 epochs suffisent sur MNIST, chaque epoch = 60 000 images

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        y_hat = model(X_batch)
        loss  = loss_fn(y_hat, y_batch)   # y_batch : (64,) — pas besoin de unsqueeze ici

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Précision sur le test set à la fin de chaque epoch
    model.eval()
    correct = 0
    with torch.no_grad():   # désactive autograd — pas besoin de gradients pour évaluer
        for X_batch, y_batch in test_loader:
            preds    = model(X_batch).argmax(dim=1)   # classe avec le score le plus haut
            correct += (preds == y_batch).sum().item()

    accuracy = correct / len(test_data)
    print(f"Epoch {epoch+1}/{EPOCHS}  —  loss: {total_loss/len(train_loader):.4f}  —  accuracy: {accuracy:.2%}")

import os; os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "./models/model.pth")
print("Modèle sauvegardé -> model.pth")
