import numpy as np

def _unbroadcast(grad, shape):
    """
    Réduit un gradient à une shape donnée en sommant les dimensions introduites
    par le broadcasting de NumPy lors de l'opération forward.

    Args:
    grad: np.ndarray, gradient à réduire, dont la shape peut différer de
    celle de la cible en raison du broadcasting.
    shape: tuple, shape du Tensor d'origine vers laquelle le gradient doit
    être ramené.

    Returns:
    np.ndarray: gradient réduit dont la shape correspond exactement à shape.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

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
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)
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
                self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def backward(self):
        """
        Calcule les gradients de tous les Tensors du graphe de calcul par rétropropagation automatique.

        Le gradient du Tensor courant est initialisé à 1, correspondant à la dérivée
        du Tensor par rapport à lui-même. L'ordre topologique du graphe est ensuite
        construit en remontant depuis ce Tensor, puis chaque noeud est parcouru dans
        l'ordre inverse afin d'exécuter sa fonction _backward() et de propager les
        gradients vers ses parents.

        Cette méthode ne prend aucun argument et ne retourne aucune valeur. Elle
        modifie directement l'attribut .grad des Tensors concernés.
        """
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        """
        Retourne l'opposé de ce Tensor 

        Returns:
            Tensor : Un nouveau Tensor qui est égal à self * -1
        """
        return self * -1 # on réutilise __mul__

    def __sub__(self, other):
        """
        Effectue une soustraction élément par élément entre ce Tensor et une autre valeur.

        Args:
            other: valeur à additionner au Tensor courant, pouvant être un Tensor,
            un scalaire, une liste ou un np.ndarray. 

            Returns:
            Tensor: nouveau Tensor correspondant au résultat de la soustraction, associé
            à la fonction permettant de propager les gradients vers les
            opérandes lors de la rétropropagation.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self + other * -1 # on réutilise __add__ et __neg__

    def __pow__(self, n):
        """
        Eleve le Tensor à la puissance n

        Args:
            n: exposant, un scalaire (int ou float), jamais un Tensor
        
        Returns:
        Tensor: nouveau Tensor qui est égal à self ** n, avec sa fonction ._backward associée
        """
        out = Tensor(self.data ** n, _children =(self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * n * self.data ** (n-1)
        out._backward = _backward
        return out

    def exp(self):
        """
        Calcule l'exponentielle de ce Tensor

        Returns:
        Tensor: nouveau Tensor qui est égal à exp(self), avec sa fonction ._backward associée
        """
        out = Tensor(np.exp(self.data), _children=(self,))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def matmul(self, other):
        """
        Effectue le produit matriciel entre ce Tensor et un autre Tensor (self @ other).

        Args:
        other: Tensor à multiplier avec le Tensor courant. Ses dimensions doivent
        être compatibles avec celles de self pour le produit matriciel,
        conformément aux règles de np.matmul. Si other n'est pas déjà
        un Tensor, il est automatiquement converti.

        Returns:
        Tensor: nouveau Tensor correspondant au résultat de self @ other,
        accompagné de sa fonction de rétropropagation associée.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data @ other.data, _children=(self, other))
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad 
        out._backward = _backward
        return out 