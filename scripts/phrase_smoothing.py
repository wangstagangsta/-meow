"""Temporal post-processing for phrase predictions: HMM/Viterbi decoding + median smoothing.

The Viterbi transition matrix is learned from labelled training sequences and bundled with the
model, so inference applies the same musically-informed smoothing the model was evaluated with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def learn_transition_matrix(labelled_df: pd.DataFrame, classes: list, smoothing: float = 1.0):
    """Count label->label transitions within each track. Row-normalized, Laplace-smoothed."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    trans = np.full((n, n), smoothing, dtype=np.float64)
    for _, tdf in labelled_df.groupby("track", sort=False):
        labels = tdf.sort_values("bar_index")["label"].tolist()
        for a, b in zip(labels[:-1], labels[1:]):
            if a in idx and b in idx:
                trans[idx[a], idx[b]] += 1
    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def viterbi_decode(proba: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Viterbi over one track's bar sequence. proba:(T,n) emission probs. Returns state idx path."""
    log_emit = np.log(proba + 1e-12)
    log_trans = np.log(trans + 1e-12)
    T, n = log_emit.shape
    dp = np.full((T, n), -np.inf)
    back = np.zeros((T, n), dtype=int)
    dp[0] = log_emit[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + log_trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0) + log_emit[t]
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def median_smooth(labels: np.ndarray, k: int = 1) -> np.ndarray:
    """Mode filter over a window of size 2k+1."""
    labels = np.asarray(labels)
    out = labels.copy()
    n = len(labels)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        vals, counts = np.unique(labels[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out
