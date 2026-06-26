"""
Train a phrase-detection model on all labelled tracks and save a standardized bundle.

Standardized bundle schema (what inference + model-swap rely on):
    model            : fitted sklearn/xgboost classifier
    label_encoder    : LabelEncoder
    spec             : FeatureSpec dict (so inference assembles features identically)
    classes          : ordered class list
    transition_matrix: (n,n) array or None (for Viterbi)
    apply_viterbi    : bool
    meta             : {model, params, bars, date, notes}

Usage:
    python scripts/train_phrase_model.py --model xgb --viterbi --name phrase_xgb_viterbi
    python scripts/train_phrase_model.py --model rf  --out models/phrase_detection/phrase_rf.joblib
    python scripts/train_phrase_model.py --model mlp --no-scalars --no-norm   # baseline-like

Feature spec defaults to the EXP-9v winning config; flags below override it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))

from phrase_features import (
    RECOMMENDED_SPEC,
    FeatureSpec,
    build_rich_dataset,
    assemble,
)
from phrase_smoothing import learn_transition_matrix

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "phrase_detection"
CACHE_PATH = MODEL_DIR / "_feature_cache.joblib"
ALL_CLASSES = ["bridge", "buildUp", "drop", "lead", "preDropFill", "quiet", "verse"]
RANDOM_STATE = 42


def make_model(name: str):
    if name == "mlp":
        def build(X, y):
            sw = compute_sample_weight(class_weight="balanced", y=y)
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                                solver="adam", learning_rate_init=1e-3, max_iter=200,
                                early_stopping=True, random_state=RANDOM_STATE)
            clf.fit(X, y, sample_weight=sw)
            return clf
        return build, {"model": "MLPClassifier", "hidden_layers": [256, 128]}
    if name == "rf":
        def build(X, y):
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                         class_weight="balanced", n_jobs=-1,
                                         random_state=RANDOM_STATE)
            clf.fit(X, y)
            return clf
        return build, {"model": "RandomForestClassifier", "n_estimators": 300}
    if name == "xgb":
        from xgboost import XGBClassifier
        def build(X, y):
            sw = compute_sample_weight(class_weight="balanced", y=y)
            clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                                n_jobs=-1, random_state=RANDOM_STATE, eval_metric="mlogloss")
            clf.fit(X, y, sample_weight=sw)
            return clf
        return build, {"model": "XGBClassifier", "n_estimators": 300, "max_depth": 6,
                       "learning_rate": 0.1}
    raise ValueError(f"unknown model '{name}'")


def get_raw_dataset(rebuild: bool):
    if CACHE_PATH.exists() and not rebuild:
        print(f"Loading cached features: {CACHE_PATH}")
        return joblib.load(CACHE_PATH)
    print("Extracting features from audio (one-time; cached afterward)...")
    df = build_rich_dataset()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(df, CACHE_PATH)
    print(f"Cached → {CACHE_PATH}")
    return df


def build_spec(args) -> FeatureSpec:
    spec = FeatureSpec(**RECOMMENDED_SPEC.to_dict())
    if args.no_scalars:
        spec.scalar_cols = []
    if args.no_norm:
        spec.per_track_norm = False
    if args.position:
        spec.add_position = True
    if args.delta:
        spec.add_delta = True
    if args.context is not None:
        spec.context_bars = args.context
    return spec


def main():
    p = argparse.ArgumentParser(description="Train a phrase-detection model bundle.")
    p.add_argument("--model", choices=["mlp", "rf", "xgb"], default="xgb")
    p.add_argument("--viterbi", action="store_true", help="bundle a Viterbi transition matrix")
    p.add_argument("--name", default=None, help="bundle name (saved under models/phrase_detection/)")
    p.add_argument("--out", default=None, help="explicit output path (overrides --name)")
    p.add_argument("--no-scalars", action="store_true", help="log-mel only (drop scalar features)")
    p.add_argument("--no-norm", action="store_true", help="disable per-track normalization")
    p.add_argument("--position", action="store_true", help="add position-in-track feature")
    p.add_argument("--delta", action="store_true", help="add delta features")
    p.add_argument("--context", type=int, default=None, help="context bars (default 8)")
    p.add_argument("--rebuild", action="store_true", help="rebuild feature cache from audio")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    spec = build_spec(args)
    print(f"Model: {args.model} | Viterbi: {args.viterbi} | Spec: {spec.to_dict()}")

    raw = get_raw_dataset(args.rebuild)
    bar_df = assemble(raw, spec)
    bar_df = bar_df[bar_df["label"] != ""].reset_index(drop=True)

    le = LabelEncoder().fit(ALL_CLASSES)
    X = np.stack(bar_df["feature"].to_list())
    y = le.transform([l if l in set(le.classes_) else le.classes_[0] for l in bar_df["label"]])
    print(f"Training on {len(bar_df)} bars, feature_dim={X.shape[1]}")

    build_fn, model_meta = make_model(args.model)
    model = build_fn(X, y)

    trans = None
    if args.viterbi:
        trans = learn_transition_matrix(bar_df, ALL_CLASSES, smoothing=1.0)

    bundle = {
        "model": model,
        "label_encoder": le,
        "spec": spec.to_dict(),
        "classes": ALL_CLASSES,
        "transition_matrix": trans,
        "apply_viterbi": args.viterbi,
        "meta": {**model_meta, "bars": len(bar_df), "feature_dim": int(X.shape[1]),
                 "date": _dt.date.today().isoformat(), "notes": args.notes},
    }

    if args.out:
        out = Path(args.out)
    else:
        name = args.name or f"phrase_{args.model}{'_viterbi' if args.viterbi else ''}"
        out = MODEL_DIR / f"{name}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"Saved bundle → {out}")


if __name__ == "__main__":
    main()
