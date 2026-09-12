import numpy as np
class Tensor:
    def __init__(self, data, _children=(), requires_grad=True):
        """
        Noeud du graphe de calcul utilisé pour effectuer l'autodifférentiation.

        Args:
        data: donnée numérique d'entrée (scalaire, liste ou np.ndarray), automatiquement convertie en np.ndarray.
        _children: tuple contenant les Tensors parents à l'origine de ce Tensor (paramètre interne utilisé lors des opérations).
        requires_grad: indique si les gradients doivent être calculés pour ce Tensor. Si False, il est exclu de la rétropropagation.

        """
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros(self.data.shape)
        self._prev = set(_children)
        self._backward = lambda: None
        self.requires_grad = requires_grad

    def __add__(self, other):
        """
        Effectue une addition élément par élément entre ce Tensor et une autre valeur.

        Args:
        other: valeur à additionner au Tensor courant, pouvant être un Tensor,
        un scalaire, une liste ou un np.ndarray. 

        Returns:
        Tensor: nouveau Tensor correspondant au résultat de l'addition, associé
        à la fonction permettant de propager les gradients vers les
        opérandes lors de la rétropropagation.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Effectue une multiplication élément par élément entre ce Tensor et une autre valeur.

        Args:
        other: valeur à multiplier au Tensor courant, pouvant être un Tensor,
        un scalaire, une liste ou un np.ndarray.

        Returns:
        Tensor: nouveau Tensor correspondant au résultat de la multiplication, associé
        à la fonction permettant de propager les gradients vers les
        opérandes lors de la rétropropagation.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other.data
            if other.requires_grad:
                other.grad += out.grad * self.data
        out._backward = _backward
        return out