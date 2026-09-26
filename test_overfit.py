"""
Test de sur-apprentissage (overfit) sur un micro-batch fixe.

Objectif : verifier que l'architecture et le pipeline d'entrainement
(forward, backward, optimizer) sont mecaniquement corrects, independamment
de la taille du modele, du corpus ou de la qualite du texte genere.

Un modele et un pipeline corrects doivent pouvoir memoriser parfaitement
un tout petit batch de donnees fixe : la loss doit s'effondrer vers 0.
Si la loss stagne ou explose, le probleme vient de l'architecture ou
du pipeline, pas de l'echelle (donnees/parametres/temps d'entrainement).
"""

import numpy as np
from gpt import GPT
from training import Adam, cross_entropy_loss

VOCAB_SIZE = 20
BATCH_SIZE = 2
SEQ_LEN = 6
N_STEPS = 300
LR = 1e-2

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_FF = 64
MAX_SEQ_LEN = 16

SEED = 0


def run_overfit_test():
    np.random.seed(SEED)

    x = np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    y = np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))

    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
    )
    optimizer = Adam(model.parameters(), lr=LR)

    print(f"Test d'overfit sur un batch fixe ({BATCH_SIZE}x{SEQ_LEN} tokens, vocab={VOCAB_SIZE})")
    print(f"{N_STEPS} iterations, lr={LR}\n")

    loss_history = []
    for step in range(N_STEPS):
        logits = model(x)
        logits_flat = logits.reshape(BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)
        y_flat = y.reshape(BATCH_SIZE * SEQ_LEN)

        loss = cross_entropy_loss(logits_flat, y_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.data))

        if step % 50 == 0 or step == N_STEPS - 1:
            print(f"step {step:3d} : loss = {loss.data:.6f}")

    final_loss = loss_history[-1]
    print()
    print(f"[PASS] loss finale = {final_loss:.6f}")
    return loss_history


if __name__ == "__main__":
    run_overfit_test()