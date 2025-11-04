import numpy as np
from collections import Counter
import math

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)
    vocab = [w for w, c in counter.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx

def compute_tfidf(all_tokens, word2idx):
    N = len(all_tokens)
    V = len(word2idx)
    tf = np.zeros((N, V))
    df = np.zeros(V)

    for i, tokens in enumerate(all_tokens):
        counts  = Counter(tokens)
        for w, c in counts.items():
            if w in word2idx:
                j = word2idx[w]
                tf[i, j] = c
                df[j] += 1
    
    idf = np.log((1+N) / (1+df)) + 1
    tfidf = tf * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf = tfidf / np.clip(norms, 1e-9, None)
    return tfidf