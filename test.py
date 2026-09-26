"""
Script d'entraînement complet du GPT from-scratch sur le corpus de Candide.
Nécessite : autodiff.py, layers.py, attention.py, transformer_block.py,
gpt.py, tokenizer.py, training.py, et data/candide.txt dans le même dossier.
"""

import time
import pickle
import numpy as np

from gpt import GPT
from tokenizer import WordTokenizer
from training import Adam, train

np.random.seed(0)

with open("data/gutenberg.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = WordTokenizer(text)
data = np.array(tokenizer.encode(text))
print(f"vocab_size = {tokenizer.vocab_size}")
print(f"corpus encodé = {len(data)} tokens")

with open("data/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

model = GPT(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    max_seq_len=32,
)
n_params = sum(p.data.size for p in model.parameters())
print(f"nombre de paramètres : {n_params:,}")

optimizer = Adam(model.parameters(), lr=3e-4)

N_STEPS = 3000
BATCH_SIZE = 8
SEQ_LEN = 32
"""
t0 = time.time()
loss_history = train(model, data, optimizer, n_steps=N_STEPS, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
t1 = time.time()

print(f"\nEntraînement terminé en {t1 - t0:.1f}s")
print(f"Loss initiale (moyenne des 20 premières) : {np.mean(loss_history[:20]):.3f}")
print(f"Loss finale   (moyenne des 20 dernières) : {np.mean(loss_history[-20:]):.3f}")

weights = [p.data.copy() for p in model.parameters()]
with open("data/model_weights.pkl", "wb") as f:
    pickle.dump(weights, f)
print("Poids sauvegardés dans data/model_weights.pkl")
"""
# Test de génération 
prompts = ["Il était une fois"]

print("\n--- Génération après entraînement ---")
for prompt in prompts:
    prompt_idx = np.array([tokenizer.encode(prompt)])
    generated = model.generate(prompt_idx, max_new_tokens=25, temp=1, top_k=10)
    print(f"\nPrompt : {prompt!r}")
    print("Suite  :", tokenizer.decode(generated[0]))

