import numpy as np
from autodiff import Tensor

class Module:
    def __init__(self):
        pass

    def parameters(self):
        """
        Retourne les Tensors entraînables correspondant aux paramètres de ce module.

        Par défaut, cette méthode retourne une liste vide. Les sous-classes possédant
        des paramètres entraînables, comme Linear, doivent redéfinir cette méthode
        afin de retourner leurs paramètres.

        Returns:
            list[Tensor]: liste des paramètres entraînables du module.
        """
        return []

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Couche linéaire (fully-connected) appliquant la transformation affine :

        y = x @ W + b

        où W représente les poids et b le biais.

        Args:
            in_features: int, dimension du vecteur d'entrée.
            out_features: int, dimension du vecteur de sortie.

        Attributes:
            W (Tensor): matrice de poids de shape (in_features, out_features),
            initialisée aléatoirement selon l'initialisation de Xavier/Glorot.
            b (Tensor): vecteur de biais de shape (out_features,), initialisé
            à zéro.

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        limite = np.sqrt(6.0 / (in_features + out_features))
        W_data = np.random.uniform(-limite, limite, size=(in_features, out_features))
        self.W = Tensor(W_data, requires_grad=True)
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        """
        Applique la transformation affine de la couche linéaire à l'entrée x :
        y = x @ W + b

        Args:
            x (Tensor): Tensor d'entrée de shape (..., in_features), par exemple
            (batch_size, in_features) ou (batch_size, seq_len, in_features).

        Returns:
            Tensor: Tensor de sortie de shape (..., out_features), correspondant
            au résultat de la transformation affine x @ W + b.
        """
        return x.matmul(self.W) + self.b 

    def parameters(self):
        """
        Retourne les paramètres entraînables de cette couche

        Returns:
            list[Tensor]: [W, b]
        """
        return [self.W, self.b]

class Embedding(Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Lookup table entre les indices de tokens et les vecteurs denses.

        Args:
            vocab_size: nombre total de tokens distincts dans le vocabulaire.
            d_model: dimension des vecteurs d'embedding.

        Attributs:
            weight (Tensor): matrice de shape (vocab_size, d_model), une ligne
            par token, initialisée selon une loi normale centrée réduite
            multipliée par 0.02 (convention GPT-2/GPT-3 pour l'initialisation
            des embeddings).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        weight_data = np.random.randn(vocab_size, d_model) * 0.02
        self.weight = Tensor(weight_data, requires_grad=True)

    def __call__(self, idx):
        """
        Retourne les vecteurs d'embedding correspondant aux indices donnés

        Args:
            idx: np.ndarray d'entiers, de shape (...) quelconque — typiquement
            (batch_size, seq_len) — contenant les indices de tokens à
            encoder. Ce n'est PAS un Tensor (juste des indices bruts)

        Returns:
            Tensor: vecteurs d'embedding de shape (..., d_model), avec le
            graphe de calcul permettant de rétropropager vers les lignes
            utilisées de self.weight
        """
        out_data = self.weight.data[idx]
        out = Tensor(out_data, _children=(self.weight,))
        def _backward():
            if self.weight.requires_grad:
                np.add.at(self.weight.grad, idx, out.grad)

        out._backward = _backward
        return out 

    def parameters(self):
        """
        Retourne les paramètres entraînables de cette couche.

        Returns:
            list[Tensor]: [weight]
        """
        return [self.weight]

class LayerNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Normalisation de couche (Normalization Layer) appliquée sur la dernière
        dimension du Tensor (d_model).

        LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

        Args:
            d_model: int, dimension du dernier axe sur lequel la normalisation
            est effectuée.
            eps: float, petite constante ajoutée à la variance afin d'éviter
            une division par zéro.

        Attributes:
            gamma (Tensor): facteur d'échelle appris de shape (d_model,),
            initialisé à 1.
            beta (Tensor): terme de décalage appris de shape (d_model,),
            initialisé à 0.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        self.gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.beta = Tensor(np.zeros(d_model), requires_grad=True)

    def __call__(self, x):
        """
        Applique la normalisation de couche à l'entrée x.

        Args:
            x: Tensor d'entrée, de shape (..., d_model), la normalisation
            est calculée le long du dernier axe uniquement.

        Returns:
            Tensor: sortie normalisée puis mise à l'échelle, de même shape
            que x.
        """
        last_axis = len(x.data.shape) - 1
        mean = x.sum(axis=last_axis, keepdims=True) / x.data.shape[last_axis]

        diff = x - mean
        var = (diff ** 2).sum(axis=last_axis, keepdims=True) / x.data.shape[last_axis]

        x_norm = diff / (var + self.eps) ** 0.5

        return self.gamma * x_norm + self.beta

    def parameters(self):
        """
        Retourne les paramètres entraînables de cette couche.

        Returns:
            list[Tensor]: [gamma, beta]
        """
        return [self.gamma, self.beta]

def gelu(x):
    """
    Applique l'activation GELU (*Gaussian Error Linear Unit*) à l'entrée,
    en utilisant l'approximation par tangente hyperbolique employée dans GPT-2.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    Args:
        x: Tensor d'entrée de shape quelconque.

    Returns:
        Tensor: résultat de GELU(x), de même shape que x
    """
    inner = (x + x**3 * 0.044715) * np.sqrt(2.0 / np.pi)
    e_pos = inner.exp()
    e_neg = (inner * -1).exp()
    tanh_inner = (e_pos - e_neg) / (e_pos + e_neg)
    return x * 0.5 * (tanh_inner + 1)

def softmax(x, axis=-1):
    """
    Calcule le softmax de x le long d'un axe donné afin de transformer des
    scores en distribution de probabilités.

    softmax(x)_i = exp(x_i) / sum_j(exp(x_j))

    Utilise une version numériquement stable en soustrayant le maximum avant
    l'exponentiation afin d'éviter les débordements.

    Args:
        x: Tensor d'entrée de shape quelconque.
        axis: int, axe du softmax (par défaut, le dernier).

    Returns:
    Tensor: distribution de probabilités de même shape que x
    """
    max_vals = np.max(x.data, axis=axis, keepdims=True)
    max_tensor = Tensor(max_vals, requires_grad=False)
    x_shifted = x - max_tensor
    exp_x = x_shifted.exp()
    sum_exp = exp_x.sum(axis=axis, keepdims=True)

    return exp_x / sum_exp