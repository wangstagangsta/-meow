"""Temporal post-processing: HMM/Viterbi decoding and median smoothing.

Viterbi uses an emission model = the classifier's per-bar probabilities, and a transition
matrix learned per-fold from the TRAINING label sequences only (no leakage). This enforces
'phrases are runs' and penalizes musically implausible transitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def learn_transition_matrix(train_df: pd.DataFrame, classes: list, smoothing: float = 1.0):
    """Count label->label transitions within each track (training data only)."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    trans = np.full((n, n), smoothing, dtype=np.float64)
    for _, tdf in train_df.groupby("track", sort=False):
        labels = tdf.sort_values("bar_index")["label"].tolist()
        for a, b in zip(labels[:-1], labels[1:]):
            if a in idx and b in idx:
                trans[idx[a], idx[b]] += 1
    trans /= trans.sum(axis=1, keepdims=True)
    return trans


def viterbi_decode(log_proba: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Standard Viterbi. log_proba: (T, n_states) emission log-probs. Returns state idx path."""
    T, n = log_proba.shape
    dp = np.full((T, n), -np.inf)
    back = np.zeros((T, n), dtype=int)
    dp[0] = log_proba[0]
    for t in range(1, T):
        # scores[i, j] = dp[t-1, i] + log_trans[i, j]
        scores = dp[t - 1][:, None] + log_trans
        back[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0) + log_proba[t]
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def median_smooth(labels: np.ndarray, k: int = 3) -> np.ndarray:
    """Mode filter over a window of size 2k+1."""
    labels = np.asarray(labels)
    out = labels.copy()
    n = len(labels)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        window = labels[lo:hi]
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out
