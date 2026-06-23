"""Model factories. All apply class-balanced weighting so the model comparison
is not confounded by imbalance handling."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42


def make_mlp():
    def build(X_train, y_train):
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train, sample_weight=sw)
        return clf
    return build, {
        "model": "MLPClassifier", "hidden_layers": (256, 128),
        "max_iter": 200, "early_stopping": True, "imbalance": "balanced_sample_weight",
    }


def make_random_forest():
    def build(X_train, y_train):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        return clf
    return build, {
        "model": "RandomForestClassifier", "n_estimators": 300,
        "min_samples_leaf": 2, "class_weight": "balanced",
    }


def make_xgboost():
    """Returns None if xgboost not installed (caller handles skip)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, None

    def build(X_train, y_train):
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        )
        clf.fit(X_train, y_train, sample_weight=sw)
        return clf
    return build, {
        "model": "XGBClassifier", "n_estimators": 300, "max_depth": 6,
        "learning_rate": 0.1, "imbalance": "balanced_sample_weight",
    }
