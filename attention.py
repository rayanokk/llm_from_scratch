import numpy as np
from autodiff import Tensor
from layers import softmax

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