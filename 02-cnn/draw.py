"""
Script interactif : dessine un chiffre, le CNN le reconnaît en temps réel.
Run : python draw.py   (nécessite que model.pth existe — lance main.py d'abord)
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tkinter as tk

from model import CNN

# ── Chargement du modèle ──────────────────────────────────────────────
model = CNN()
model.load_state_dict(torch.load("models/model.pth", weights_only=True))
model.eval()

# ── Preprocessing — même pipeline que l'entraînement ─────────────────
def preprocess(pil_image):
    img = pil_image.resize((28, 28), Image.LANCZOS).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081               # même Normalize que train
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor

def predict(pil_image):
    with torch.no_grad():
        logits = model(preprocess(pil_image))
        probs  = torch.softmax(logits, dim=1)[0]
        digit  = probs.argmax().item()
        conf   = probs[digit].item()
    return digit, conf, probs


# ── Interface ─────────────────────────────────────────────────────────
CANVAS_SIZE = 280   # 280px affiché, réduit à 28px pour la prédiction
PEN_SIZE    = 16

root = tk.Tk()
root.title("Dessine un chiffre")
root.configure(bg="#0f0f1a")
root.resizable(False, False)

# Canvas de dessin
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                   bg="black", cursor="crosshair", highlightthickness=0)
canvas.pack(padx=16, pady=(16, 8))

# Image PIL en arrière-plan (c'est ce qu'on envoie au modèle)
pil_img  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
pil_draw = ImageDraw.Draw(pil_img)

# Zone résultat
result_var = tk.StringVar(value="Dessine un chiffre...")
result_lbl = tk.Label(root, textvariable=result_var, bg="#0f0f1a",
                      fg="white", font=("Helvetica", 22, "bold"))
result_lbl.pack(pady=(0, 4))

# Barres de confiance pour chaque classe
BAR_W, BAR_H = 280, 16
bar_frame = tk.Frame(root, bg="#0f0f1a")
bar_frame.pack(pady=(0, 16))

bar_labels  = []
bar_canvases = []
for i in range(10):
    row = tk.Frame(bar_frame, bg="#0f0f1a")
    row.pack(fill="x", pady=1)
    tk.Label(row, text=str(i), bg="#0f0f1a", fg="#888888",
             font=("Helvetica", 10), width=2).pack(side="left")
    bc = tk.Canvas(row, width=BAR_W, height=BAR_H, bg="#13132b",
                   highlightthickness=0)
    bc.pack(side="left", padx=4)
    lbl = tk.Label(row, text="", bg="#0f0f1a", fg="#888888",
                   font=("Helvetica", 9), width=5)
    lbl.pack(side="left")
    bar_canvases.append(bc)
    bar_labels.append(lbl)

# Bouton effacer
def clear():
    canvas.delete("all")
    pil_draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
    result_var.set("Dessine un chiffre...")
    for bc in bar_canvases:
        bc.delete("all")
    for lbl in bar_labels:
        lbl.config(text="")

tk.Button(root, text="Effacer", command=clear,
          bg="#1d3557", fg="white", font=("Helvetica", 11),
          relief="flat", padx=16, pady=4).pack(pady=(0, 16))


# ── Dessin + prédiction en temps réel ────────────────────────────────
last_x, last_y = None, None

def on_draw(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x and last_y:
        canvas.create_line(last_x, last_y, x, y,
                           fill="white", width=PEN_SIZE,
                           capstyle=tk.ROUND, smooth=True)
        pil_draw.line([last_x, last_y, x, y], fill=255, width=PEN_SIZE)
    last_x, last_y = x, y
    update_prediction()

def on_release(event):
    global last_x, last_y
    last_x, last_y = None, None

def update_prediction():
    # Légère gaussienne pour lisser le trait avant de prédire
    smoothed = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    digit, conf, probs = predict(smoothed)

    result_var.set(f"→  {digit}   ({conf:.0%})")

    for i, bc in enumerate(bar_canvases):
        bc.delete("all")
        p = probs[i].item()
        w = int(BAR_W * p)
        color = "#00ff99" if i == digit else "#1d3557"
        if w > 0:
            bc.create_rectangle(0, 0, w, BAR_H, fill=color, outline="")
        bar_labels[i].config(text=f"{p:.0%}")

canvas.bind("<B1-Motion>", on_draw)
canvas.bind("<ButtonRelease-1>", on_release)

root.mainloop()
