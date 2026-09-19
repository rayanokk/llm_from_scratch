from autodiff import Tensor
import numpy as np
import tqdm

def cross_entropy_loss(logits, targets):
    """
    Calcule la cross entropy entre des logits prédits et les vrais indices 
    de tokens attendus
    Args:
        logits: Tensor de shape (N, vocab_size), les scores bruts prédits
        pour N exemples.
        targets: np.ndarray d'entiers de shape (N,), l'indice du vrai
        token attendu pour chaque exemple.

    Returns:
        Tensor: scalaire (shape ()), la perte moyenne sur les N exemples.
    """
    max_tensor = Tensor(
        np.max(logits.data, axis=-1, keepdims=True),
        requires_grad=False
    )
    shifted = logits - max_tensor
    log_sum_exp = shifted.exp().sum(axis=-1, keepdims=True).log()
    log_probs = shifted - log_sum_exp

    N = logits.data.shape[0]
    selected = log_probs[np.arange(N), targets].sum() * -1 / N

    return selected

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Optimizer Adam (Kingma & Ba, 2014), implémenté à partir des gradients
        accumulés dans les Tensors par backward().

        Args:
            params: list[Tensor], les paramètres à optimiser.
            lr: float, taux d'apprentissage.
            beta1: float, coefficient de décroissance du premier moment
            (moyenne mobile du gradient).
            beta2: float, coefficient de décroissance du second moment
            (moyenne mobile du gradient au carré).
            eps: float, constante de stabilité numérique évitant une division
            par zéro.

        Attributs:
            t (int): compteur d'itérations, initialisé à 0.
            m (list[np.ndarray]): premier moment, un array par paramètre,
            initialisé à zéro.
            v (list[np.ndarray]): second moment, un array par paramètre,
            initialisé à zéro.
        """
        self.params = params
        self.lr = lr 
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.eps = eps 
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        """
        Effectue une mise à jour Adam de tous les paramètres, en utilisant
        leurs gradients actuels (.grad)

        Ne prend aucun argument et ne retourne rien, modifie juste .data de chaque
        paramètre in-place.
        """
        self.t += 1 
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        Remet à zéro le gradient de tous les paramètres optimisés, à appeler
        avant chaque nouveau backward() (sinon les gradients de différentes
        itérations s'accumulent)
        """
        for p in self.params:
            p.grad = np.zeros_like(p.data)

def get_batch(data, batch_size, seq_len):
    """
    Échantillonne un batch de séquences (entrée, cible) à partir d'un
    corpus de tokens encodés

    Args:
        data: np.ndarray d'entiers, shape (corpus_len,), le corpus complet
            encodé en indices de tokens.
        batch_size: int, nombre de séquences dans le batch.
        seq_len: int, longueur de chaque séquence.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x, y), chacun de shape
        (batch_size, seq_len). y est x décalé d'une position
    """
    starts = np.random.randint(0, len(data) - seq_len, size=batch_size)

    x_batch = []
    y_batch = []

    for start in starts:
        x = data[start:start + seq_len]
        y = data[start + 1:start + seq_len + 1]
        x_batch.append(x)
        y_batch.append(y)

    x = np.stack(x_batch)
    y = np.stack(y_batch)

    return x, y

def train(model, data, optimizer, n_steps, batch_size, seq_len):
    """
    Boucle d'entraînement principale : échantillonne des batches,
    calcule la loss, rétropropage, met à jour les paramètres

    Args:
        model: GPT, le modèle à entraîner.
        data: np.ndarray d'entiers, shape (corpus_len,), le corpus complet
        encodé en indices de tokens.
        optimizer: Adam, l'optimizer déjà construit avec model.parameters().
        n_steps: int, nombre d'itérations d'entraînement.
        batch_size: int, taille du batch.
        seq_len: int, longueur de séquence.

    Returns:
        list[float]: l'historique des valeurs de loss, une par itération.
    """
    for step in tqdm(range(n_steps)):
        x, y = get_batch(data, batch_size, seq_len)
        logits = model(x)
        logits_flat = logits.reshape(batch_size*seq_len, model.vocab_size)
        y_flat = y.reshape(batch_size * seq_len)

        loss = cross_entropy_loss(logits_flat, y_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()