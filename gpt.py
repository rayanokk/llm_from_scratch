import numpy as np
from layers import Module, Embedding, LayerNorm
from attention import causal_mask
from transformer_block import TransformerBlock

class GPT(Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        """
        Modèle GPT complet (decoder-only transformer), architecture GPT-2/3.

        Args:
            vocab_size: taille du vocabulaire.
            d_model: dimension du modèle.
            n_heads: nombre de têtes d'attention par bloc.
            n_layers: nombre de blocs transformer empilés.
            d_ff: dimension intermédiaire du feed-forward de chaque bloc.
            max_seq_len: longueur maximale de séquence supportée (détermine
            la taille de la table d'embedding de position).

        Attributs:
            token_emb (Embedding): table d'embedding de tokens.
            pos_emb (Embedding): table d'embedding de position.
            blocks (list[TransformerBlock]): les n_layers blocs transformer.
            ln_final (LayerNorm): normalisation finale avant projection vocabulaire.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_seq_len, d_model)
        self.blocks : list[TransformerBlock] = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)

    def __call__(self, idx):
        """
        Calcule les logits pour chaque position de la séquence d'entrée.

        Args:
            idx: np.ndarray d'entiers, shape (batch_size, seq_len), les indices
            de tokens en entrée.

        Returns:
            Tensor: logits de shape (batch_size, seq_len, vocab_size).
        """
        _, seq_len = idx.shape
        token_embeddings = self.token_emb(idx) # (batch_size, seq_len, d_model)
        positions = np.arange(seq_len)
        position_embeddings = self.pos_emb(positions) # (seq_len, d_model)
        x = token_embeddings + position_embeddings

        mask = causal_mask(seq_len)
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_final(x)
        logits = x.matmul(self.token_emb.weight.T) # (batch_size, seq_len, vocab_size)

        return logits

    def parameters(self):
        parameters = self.token_emb.parameters() + self.pos_emb.parameters()
        for block in self.blocks:
            parameters = parameters + block.parameters()
        return parameters

    def generate(self, idx, max_new_tokens, temp=1, top_k=None):
        """
        Génère une séquence de tokens de manière autoregressive à partir d'un
        contexte initial, sans calculer de gradients.

        Args:
            idx: np.ndarray d'entiers de shape (batch_size, seq_len), contenant
            le contexte initial.
            max_new_tokens: int, nombre de nouveaux tokens à générer.
            temp: float, facteur de température appliqué aux logits avant
            le softmax. Une valeur inférieure à 1 rend la génération plus
            déterministe, tandis qu'une valeur supérieure à 1 la rend plus
            aléatoire.
            top_k: int ou None, si fourni, limite l'échantillonnage aux k tokens
            ayant les logits les plus élevés à chaque étape.

        Returns:
            np.ndarray: séquence complète contenant le contexte initial et les
            tokens générés, de shape (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_truncated = idx[:, -self.max_seq_len:]
            logits = self(idx_truncated)
            logits = logits[:, -1, :].data #(batch_size, vocab_size)
            logits = logits / temp

            if top_k is not None:
                indices = np.argsort(logits, axis=-1)
                indices_to_rmv = indices[:, :-top_k]
                logits[np.arange(logits.shape[0])[:, None], indices_to_rmv] = -np.inf
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)

            next_tokens = np.array([
                np.random.choice(logits.shape[1], p=probs[i])
                for i in range(logits.shape[0])
            ])

            next_tokens = next_tokens[:, None]
            idx = np.concatenate([idx, next_tokens], axis=-1)

        return idx