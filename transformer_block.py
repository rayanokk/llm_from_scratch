from layers import Module, Linear,LayerNorm, gelu
from attention import MultiHeadAttention

class FeedForward(Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Réseau feed-forward position-wise (Vaswani et al., section 3.3) :
        FFN(x) = Linear2(GELU(Linear1(x))).

        Args:
            d_model: dimension du modèle (entrée et sortie).
            d_ff: dimension intermédiaire (généralement 4 * d_model).

        Attributs:
            linear1 (Linear): projection de d_model vers d_ff.
            linear2 (Linear): projection de d_ff vers d_model.
        """
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def __call__(self, x):
        return self.linear2(gelu(self.linear1(x)))

    def parameters(self):
        return self.linear2.parameters() + self.linear1.parameters()

class TransformerBlock(Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """
        Bloc transformer complet, convention pré-LayerNorm (GPT-2/3) :

        x = x + MultiHeadAttention(LayerNorm1(x))
        x = x + FeedForward(LayerNorm2(x))

        Args:
            d_model: dimension du modèle.
            n_heads: nombre de têtes d'attention.
            d_ff: dimension intermédiaire du feed-forward.

        Attributs:
            attn (MultiHeadAttention): sous-couche d'attention multi-têtes.
            ffn (FeedForward): sous-couche feed-forward.
            ln1 (LayerNorm): normalisation appliquée avant l'attention.
            ln2 (LayerNorm): normalisation appliquée avant le feed-forward.
        """
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))

        return x

    def parameters(self):
        return self.attn.parameters() + self.ffn.parameters() + self.ln1.parameters() + self.ln2.parameters()

