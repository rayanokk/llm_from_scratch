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
    