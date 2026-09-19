import numpy as np

def _tokenize_words(text):
    """
    Découpe un texte en une liste de tokens

    Args:
        text: str à découper

    Returns:
        list[str]: liste ordonnée des tokens (mots et ponctuation).
    """
    tokens = []
    current_word = ""
    for ch in text:
        if ch.isalnum():
            current_word += ch
        else:
            if current_word != "":
                tokens.append(current_word)
                current_word = ""
            if not ch.isspace():
                tokens.append(ch)
    if current_word != "":
        tokens.append(current_word)

    return tokens

class WordTokenizer:
    def __init__(self, text: str):
        """
        Tokenizer word-level : découpe le texte en mots, chaque mot/symbole unique devenant un token.
        Inclut un token spécial <UNK> pour les mots hors-vocabulaire lors
        de l'encodage.

        Args:
            text: str, corpus de référence utilisé pour construire le
            vocabulaire.

        Attributs:
            words (list[str]): liste triée des tokens uniques, <UNK> inclus.
            vocab_size (int): nombre de tokens distincts, len(words).
            word_to_idx (dict[str, int]): mot -> indice.
            idx_to_word (dict[int, str]): indice -> mot.
        """
        self.words = sorted(set(_tokenize_words(text)) | {"<UNK>"})
        self.vocab_size = len(self.words)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for i, w in enumerate(self.words)}

    def encode(self, text: str):
        """
        Convertit une chaîne de caractères en liste d'indices de tokens.
        Les mots absents du vocabulaire sont remplacés par le token <UNK>.

        Args:
            text: str à encoder.

        Returns:
            list[int]: indices correspondant à chaque mot/symbole de text.
        """
        tokens = _tokenize_words(text)
        idx = [self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in tokens]

        return idx

    def decode(self, idx):
        """
        Convertit une liste (ou np.ndarray) d'indices de tokens en texte

        Args:
            indices: liste ou np.ndarray d'entiers, indices de tokens.

        Returns:
            str: texte reconstruit, mots séparés par des espaces.
        """
        tokens = []
        for i in idx:
            tokens.append(self.idx_to_word[i])
        seq = tokens[0]
        for token in tokens[1:]:
            seq = seq + " " + token

        return seq

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
    x_batch = []
    y_batch = []
    starts = np.random.randint(0, len(data) - seq_len, size=batch_size)
    for start in starts:
        x = data[start:start+seq_len]
        y = data[start+1:start+seq_len+1]
        x_batch.append(x)
        y_batch.append(y)
    x = np.stack(x_batch)
    y = np.stack(y_batch)

    return x, y