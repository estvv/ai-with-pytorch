"""
Complétion de code en temps réel — le RNN continue ce que tu écris.
Run : python complete.py   (nécessite models/model.pth)
"""
import torch
import tkinter as tk
from model import RNN

# ── Chargement ────────────────────────────────────────────────────────
checkpoint = torch.load("models/model.pth", weights_only=False)
chars      = checkpoint["vocab"]
c2i        = {c: i for i, c in enumerate(chars)}
i2c        = {i: c for i, c in enumerate(chars)}

model = RNN(len(chars), checkpoint["embed_dim"],
            checkpoint["hidden_dim"], checkpoint["n_layers"])
model.load_state_dict(checkpoint["model_state"])
model.eval()


# ── Génération ────────────────────────────────────────────────────────
def generate(seed, length=200, temperature=0.7):
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


# ── Interface ─────────────────────────────────────────────────────────
BG, AX_BG = "#0f0f1a", "#13132b"
FONT_CODE  = ("Courier New", 12)

root = tk.Tk()
root.title("RNN — complétion de code")
root.configure(bg=BG)
root.resizable(True, True)
root.geometry("820x600")

# Label
tk.Label(root, text="Tape du code C/C++ — le RNN complète automatiquement",
         bg=BG, fg="#888888", font=("Helvetica", 10)).pack(pady=(12, 4))

# Frame principale avec les deux zones côte à côte
frame = tk.Frame(root, bg=BG)
frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))
frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)
frame.rowconfigure(1, weight=1)

tk.Label(frame, text="Ton code", bg=BG, fg="#7ec8e3",
         font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
tk.Label(frame, text="Complétion du RNN", bg=BG, fg="#ffd166",
         font=("Helvetica", 10, "bold")).grid(row=0, column=1, sticky="w",
                                               padx=(16, 0), pady=(0, 4))

# Zone de saisie
input_text = tk.Text(frame, bg=AX_BG, fg="white", insertbackground="white",
                     font=FONT_CODE, relief="flat", wrap="none",
                     highlightthickness=1, highlightbackground="#1e1e3a")
input_text.grid(row=1, column=0, sticky="nsew")

# Scrollbar partagée
scrollbar = tk.Scrollbar(frame, bg=BG, troughcolor=AX_BG)
scrollbar.grid(row=1, column=2, sticky="ns")

# Zone de complétion (lecture seule)
output_text = tk.Text(frame, bg="#0a0a14", fg="#ffd166", insertbackground="white",
                      font=FONT_CODE, relief="flat", wrap="none",
                      state="disabled", highlightthickness=1,
                      highlightbackground="#1e1e3a",
                      yscrollcommand=scrollbar.set)
output_text.grid(row=1, column=1, sticky="nsew", padx=(16, 0))
scrollbar.config(command=output_text.yview)

# Barre de statut
status_var = tk.StringVar(value="En attente...")
status_lbl = tk.Label(root, textvariable=status_var, bg=BG, fg="#555555",
                      font=("Helvetica", 9), anchor="w")
status_lbl.pack(fill="x", padx=16, pady=(0, 8))

# Temperature slider
ctrl_frame = tk.Frame(root, bg=BG)
ctrl_frame.pack(fill="x", padx=16, pady=(0, 12))
tk.Label(ctrl_frame, text="Créativité :", bg=BG, fg="#888888",
         font=("Helvetica", 9)).pack(side="left")
temp_var = tk.DoubleVar(value=0.7)
tk.Scale(ctrl_frame, variable=temp_var, from_=0.2, to=1.5, resolution=0.1,
         orient="horizontal", bg=BG, fg="white", troughcolor=AX_BG,
         highlightthickness=0, length=160, showvalue=True,
         font=("Helvetica", 8)).pack(side="left", padx=(4, 16))
tk.Label(ctrl_frame, text="0.2 = conservateur  ·  1.5 = créatif",
         bg=BG, fg="#555555", font=("Helvetica", 8)).pack(side="left")

# ── Logique de complétion ─────────────────────────────────────────────
_after_id = None

def on_keyup(event):
    global _after_id
    if _after_id:
        root.after_cancel(_after_id)
    # Délai de 400ms après la dernière frappe pour éviter de générer à chaque caractère
    _after_id = root.after(400, do_complete)

def do_complete():
    seed = input_text.get("1.0", "end-1c")
    if not seed.strip():
        output_text.config(state="normal")
        output_text.delete("1.0", "end")
        output_text.config(state="disabled")
        status_var.set("En attente...")
        return

    status_var.set("Génération en cours...")
    root.update()

    completion = generate(seed[-200:], length=200, temperature=temp_var.get())

    output_text.config(state="normal")
    output_text.delete("1.0", "end")
    output_text.insert("1.0", completion)
    output_text.config(state="disabled")
    status_var.set(f"Complété ({len(completion)} caractères générés)")

input_text.bind("<KeyRelease>", on_keyup)

# Texte de départ
input_text.insert("1.0", "int\t")
do_complete()

root.mainloop()
