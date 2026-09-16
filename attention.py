import numpy as np
from autodiff import Tensor
from layers import softmax, Module, Linear

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calcule le mécanisme d'attention par produit scalaire mis à l'échelle
    (Scaled Dot-Product Attention, Vaswani et al., section 3.2.1) :

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k) + mask) @ V

    Args:
    Q: Tensor des Queries, de shape (seq_len, d_k).
    K: Tensor des Keys, de shape (seq_len, d_k).
    V: Tensor des Values, de shape (seq_len, d_v).
    mask: Tensor optionnel de shape (seq_len, seq_len), contenant 0 pour
    les positions autorisées et une grande valeur négative pour les
    positions masquées. `None` si aucun masquage n'est appliqué.

    Returns:
    Tensor: résultat de l'attention, de shape (seq_len, d_v)
    """
    d_k = K.data.shape[-1]
    scores = Q.matmul(K.T) # (seq_len, seq_len)
    scores = scores / (d_k ** 0.5)
    if mask is not None:
        scores = scores + mask
    attention = softmax(scores, axis=-1)

    return attention.matmul(V) 

def causal_mask(seq_len):
    """
    Construit un masque causal empêchant chaque position d'accéder aux positions
    futures, comme dans les modèles decoder-only tels que GPT

    Args:
        seq_len: int, longueur de la séquence.

    Returns:
        Tensor: masque de shape (seq_len, seq_len) avec requires_grad=False.

    """
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    zeros = np.zeros((seq_len, seq_len))
    zeros[mask] = -1e9
    mask_tensor = Tensor(zeros, requires_grad=False)

    return mask_tensor

class MultiHeadAttention(Module):
    def __init__(self, d_model: int, n_heads: int):
        """
        Attention multi-têtes (Vaswani et al., section 3.2.2) : applique
        l'attention en parallèle sur n_heads sous-espaces de dimension d_k,
        puis recombine les résultats.

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        où head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

        Args:
            d_model: dimension du modèle (dimension des embeddings/résidus).
            n_heads: nombre de têtes d'attention. 
        Attributs:
            d_k (int): dimension par tête, d_model // n_heads
            W_Q, W_K, W_V (Linear): projections Query/Key/Value, chacune
            de d_model vers d_model
            W_O (Linear): projection de sortie, de d_model vers d_model
        """
        super().__init__()
        self.d_model = d_model 
        self.n_heads = n_heads 
        self.d_k = d_model // n_heads 
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

    def __call__(self, x, mask=None):
        """
        Applique le mécanisme d'attention multi-têtes
        à l'entrée x en self-attention, où les Queries, Keys et Values sont
        toutes dérivées de x.

        Args:
            x: Tensor d'entrée de shape (batch_size, seq_len, d_model).
            mask: Tensor optionnel de masquage, broadcastable avec la shape
            (batch_size, n_heads, seq_len, seq_len).

        Returns:
            Tensor: sortie de l'attention multi-têtes de shape
            (batch_size, seq_len, d_model).

        """
        batch_size, seq_len, _ = x.data.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        Q = Q.transpose((0, 2, 1, 3))
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.transpose((0, 2, 1, 3))
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.transpose((0, 2, 1, 3))

        scores = Q.matmul(K.transpose((0, 1, 3, 2))) / self.d_k**0.5 #(batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores + mask 
        attn = softmax(scores, axis=-1)
        attn = attn.matmul(V) # (batch_size, n_heads, seq_len, d_k)
        attn = attn.transpose((0, 2, 1, 3))  # (batch_size, seq_len, n_heads, d_k)
        attn = attn.reshape(batch_size, seq_len, self.d_model)

        return self.W_O(attn)

    def parameters(self):
        return self.W_K.parameters() + self.W_Q.parameters() + self.W_V.parameters() + self.W_O.parameters()



        


    