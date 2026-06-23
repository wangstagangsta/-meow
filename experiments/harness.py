"""
Shared evaluation harness for phrase detection experiments.
All functions here are read-only with respect to the repo; they only write inside experiments/.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import KFold

EXPERIMENTS_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENTS_DIR.parent
DATA_DIR = REPO_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
LABEL_DIR = DATA_DIR / "labels"
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"
FOLDS_PATH = EXPERIMENTS_DIR / "folds.json"
STATE_PATH = EXPERIMENTS_DIR / "state.json"
RESULTS_PATH = EXPERIMENTS_DIR / "results.md"
LOG_PATH = EXPERIMENTS_DIR / "log.jsonl"
INSIGHTS_PATH = EXPERIMENTS_DIR / "insights.md"
BLOCKED_PATH = EXPERIMENTS_DIR / "blocked.md"
ERRORS_PATH = EXPERIMENTS_DIR / "errors.log"

N_FOLDS = 5
FOLD_SEED = 2024
ALL_CLASSES = ["bridge", "buildUp", "drop", "lead", "preDropFill", "quiet", "verse"]
KEY_CLASSES = ["drop", "buildUp", "verse"]

# ---------------------------------------------------------------------------
# Fold management
# ---------------------------------------------------------------------------

def build_or_load_folds(tracks: List[str]) -> Dict[str, int]:
    """Freeze track→fold mapping once, then always load from disk."""
    if FOLDS_PATH.exists():
        mapping = json.loads(FOLDS_PATH.read_text())
        # Validate all tracks are present
        missing = [t for t in tracks if t not in mapping]
        if missing:
            log_insight(f"WARNING: {len(missing)} tracks not in existing folds.json — were tracks added? Missing: {missing}")
        return mapping

    tracks_sorted = sorted(tracks)
    rng = np.random.default_rng(FOLD_SEED)
    shuffled = rng.permuted(tracks_sorted).tolist()
    kf = KFold(n_splits=N_FOLDS, shuffle=False)
    mapping = {}
    for fold_idx, (_, val_idx) in enumerate(kf.split(shuffled)):
        for i in val_idx:
            mapping[shuffled[i]] = fold_idx

    FOLDS_PATH.write_text(json.dumps(mapping, indent=2))
    log_insight(f"Froze {N_FOLDS}-fold split ({FOLD_SEED=}) over {len(tracks)} tracks → folds.json")
    return mapping


def fold_assignments(bar_df: pd.DataFrame, fold_map: Dict[str, int]) -> pd.Series:
    return bar_df["track"].map(fold_map)


# ---------------------------------------------------------------------------
# Feature helpers (imported pipeline will provide extract; this wraps it)
# ---------------------------------------------------------------------------

def df_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack(df["feature"].to_list())
    y = df["label"].to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Boundary F1 and drop-onset metrics
# ---------------------------------------------------------------------------

def compute_boundaries(labels: np.ndarray) -> np.ndarray:
    """Return bar indices where the phrase changes (i.e. boundary bars)."""
    labels = np.asarray(labels)
    boundaries = np.where(labels[:-1] != labels[1:])[0] + 1
    return boundaries


def boundary_f1(true_labels: np.ndarray, pred_labels: np.ndarray, tol: int = 1) -> float:
    """
    Compute F1 for phrase boundaries with ±tol bar tolerance.
    Each true boundary can be matched at most once (greedy nearest-first).
    """
    true_b = compute_boundaries(true_labels)
    pred_b = compute_boundaries(pred_labels)

    if len(true_b) == 0 and len(pred_b) == 0:
        return 1.0
    if len(true_b) == 0 or len(pred_b) == 0:
        return 0.0

    matched_true = set()
    matched_pred = set()
    # Greedy: sort candidate pairs by distance
    pairs = sorted(
        [(abs(int(t) - int(p)), ti, pi)
         for ti, t in enumerate(true_b)
         for pi, p in enumerate(pred_b)
         if abs(int(t) - int(p)) <= tol],
        key=lambda x: x[0],
    )
    for _, ti, pi in pairs:
        if ti not in matched_true and pi not in matched_pred:
            matched_true.add(ti)
            matched_pred.add(pi)

    tp = len(matched_true)
    precision = tp / len(pred_b) if pred_b.size > 0 else 0.0
    recall = tp / len(true_b) if true_b.size > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def drop_onset_error(true_labels: np.ndarray, pred_labels: np.ndarray) -> Optional[float]:
    """
    Median |predicted - true| bars for the first 'drop' boundary in each track.
    Returns None if no drop boundary exists in the ground truth.
    """
    true_drop_idxs = np.where(
        (np.array(list(true_labels[:-1])) != "drop") &
        (np.array(list(true_labels[1:])) == "drop")
    )[0] + 1
    pred_drop_idxs = np.where(
        (np.array(list(pred_labels[:-1])) != "drop") &
        (np.array(list(pred_labels[1:])) == "drop")
    )[0] + 1

    if len(true_drop_idxs) == 0:
        return None
    true_first = true_drop_idxs[0]
    pred_first = pred_drop_idxs[0] if len(pred_drop_idxs) > 0 else None
    if pred_first is None:
        return None
    return float(abs(int(true_first) - int(pred_first)))


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def run_cv(
    bar_df: pd.DataFrame,
    fold_map: Dict[str, int],
    build_model_fn,
    label_classes: List[str],
    fit_scaler_fn=None,
    train_hmm_fn=None,
    smoother=None,
) -> Dict:
    """
    Run k-fold CV.
    - build_model_fn(X_train, y_train) -> fitted model
    - fit_scaler_fn(X_train) -> (scaler_or_None, X_train_scaled) [optional, per-fold]
    - train_hmm_fn(train_df, classes) -> hmm/transition obj [optional, per-fold, train only]
    - smoother(proba_track, classes, hmm) -> per-track predicted label-index array [optional]
        Applied per contiguous track sequence. If None, argmax per bar.
    Returns dict of per-fold + aggregated metrics.
    """
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    label_encoder.fit(label_classes)

    fold_results = []
    fold_ids = sorted(set(fold_map.values()))

    for fold in fold_ids:
        val_tracks = [t for t, f in fold_map.items() if f == fold]
        train_tracks = [t for t, f in fold_map.items() if f != fold]

        val_df = bar_df[bar_df["track"].isin(val_tracks)].copy()
        train_df = bar_df[bar_df["track"].isin(train_tracks)].copy()

        # Drop unlabeled bars, sort so per-track sequences are contiguous + ordered
        train_df = train_df[train_df["label"] != ""].sort_values(["track", "bar_index"]).reset_index(drop=True)
        val_df = val_df[val_df["label"] != ""].sort_values(["track", "bar_index"]).reset_index(drop=True)

        if len(val_df) == 0 or len(train_df) == 0:
            continue

        X_train, y_train_str = df_to_arrays(train_df)
        X_val, y_val_str = df_to_arrays(val_df)

        # Per-fold scaler (no leakage)
        if fit_scaler_fn is not None:
            scaler, X_train = fit_scaler_fn(X_train)
            if scaler is not None:
                X_val = scaler.transform(X_val)

        # Encode labels, handling unseen classes safely
        known = set(label_encoder.classes_)
        y_train_enc = label_encoder.transform(
            [l if l in known else label_encoder.classes_[0] for l in y_train_str]
        )

        t0 = time.perf_counter()
        model = build_model_fn(X_train, y_train_enc)
        train_time = time.perf_counter() - t0

        # Align model's class order to label_encoder indices
        t0 = time.perf_counter()
        proba_raw = model.predict_proba(X_val)
        infer_time = time.perf_counter() - t0
        # Map model.classes_ (encoded ints) to columns over full label set
        proba = np.zeros((proba_raw.shape[0], len(label_classes)), dtype=np.float64)
        for col, cls_enc in enumerate(model.classes_):
            proba[:, int(cls_enc)] = proba_raw[:, col]

        # HMM/transition learned per-fold from TRAIN only
        hmm = train_hmm_fn(train_df, label_classes) if train_hmm_fn is not None else None

        # Predict per-track (so smoothing sees contiguous sequences)
        y_pred_enc = np.empty(len(val_df), dtype=int)
        for track, tdf in val_df.groupby("track", sort=False):
            rows = tdf.index.to_numpy()
            p = proba[rows]
            if smoother is not None:
                y_pred_enc[rows] = smoother(p, label_classes, hmm)
            else:
                y_pred_enc[rows] = np.argmax(p, axis=1)

        y_pred_str = label_encoder.inverse_transform(y_pred_enc)

        # Metrics
        valid_classes = [c for c in label_classes if c in set(y_val_str)]
        macro_f1 = f1_score(y_val_str, y_pred_str, average="macro",
                            labels=valid_classes, zero_division=0)
        per_class_scores = f1_score(
            y_val_str, y_pred_str, labels=label_classes,
            average=None, zero_division=0
        )
        per_class = {c: float(per_class_scores[i]) for i, c in enumerate(label_classes)}
        verse_support = int((np.array(y_val_str) == "verse").sum())

        # Boundary metrics (per track in val fold, then average)
        b_f1s, drop_errors = [], []
        for track, tdf in val_df.groupby("track", sort=False):
            if len(tdf) < 2:
                continue
            rows = tdf.index.to_numpy()
            y_true_t = np.array(val_df.loc[rows, "label"].tolist())
            y_pred_t = y_pred_str[rows]
            b_f1s.append(boundary_f1(y_true_t, y_pred_t))
            err = drop_onset_error(y_true_t, y_pred_t)
            if err is not None:
                drop_errors.append(err)

        fold_results.append({
            "fold": fold,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "boundary_f1": float(np.mean(b_f1s)) if b_f1s else None,
            "drop_onset_error": float(np.median(drop_errors)) if drop_errors else None,
            "verse_support": verse_support,
            "train_time": train_time,
            "infer_time": infer_time,
            "val_tracks": val_tracks,
            "n_train_bars": len(train_df),
            "n_val_bars": len(val_df),
            "confusion_matrix": confusion_matrix(
                y_val_str, y_pred_str,
                labels=[c for c in label_classes if c in set(y_val_str) | set(y_pred_str)]
            ).tolist(),
            "cm_labels": [c for c in label_classes if c in set(y_val_str) | set(y_pred_str)],
        })

    # Aggregate
    macro_f1s = [r["macro_f1"] for r in fold_results]
    b_f1s_all = [r["boundary_f1"] for r in fold_results if r["boundary_f1"] is not None]
    drop_errs = [r["drop_onset_error"] for r in fold_results if r["drop_onset_error"] is not None]

    agg = {
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "macro_f1_per_fold": [float(x) for x in macro_f1s],
        "boundary_f1_mean": float(np.mean(b_f1s_all)) if b_f1s_all else None,
        "boundary_f1_std": float(np.std(b_f1s_all)) if b_f1s_all else None,
        "drop_onset_error_median": float(np.median(drop_errs)) if drop_errs else None,
        "per_class_mean": {
            c: float(np.mean([r["per_class"][c] for r in fold_results]))
            for c in label_classes
        },
        "verse_support_per_fold": [r["verse_support"] for r in fold_results],
        "train_time_total": float(sum(r["train_time"] for r in fold_results)),
        "infer_time_total": float(sum(r["infer_time"] for r in fold_results)),
        "fold_results": fold_results,
    }
    return agg


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_result(exp_id: str, exp_name: str, config: dict, metrics: dict,
               hypothesis: str, decision: str, decision_reason: str,
               notes: str = "", wall_time: float = 0.0):
    """Append one experiment block to results.md and log.jsonl."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    m = metrics

    pc = m.get("per_class_mean", {})
    fold_f1s = m.get("macro_f1_per_fold", [])
    verse_support = m.get("verse_support_per_fold", [])

    block = f"""
## {exp_id}: {exp_name}
- **timestamp:** {ts}  |  **wall time:** {wall_time:.1f}s
- **hypothesis:** {hypothesis}
- **config:**
```
{json.dumps(config, indent=2)}
```
- **metrics:**
  - Macro F1: {m.get('macro_f1_mean', '?'):.4f} ± {m.get('macro_f1_std', '?'):.4f}
  - Per-fold F1: {[f'{x:.3f}' for x in fold_f1s]}
  - Boundary F1: {f"{m['boundary_f1_mean']:.4f} ± {m['boundary_f1_std']:.4f}" if m.get('boundary_f1_mean') is not None else 'N/A'}
  - Drop-onset error (bars): {f"{m['drop_onset_error_median']:.1f}" if m.get('drop_onset_error_median') is not None else 'N/A'}
  - Per-class F1 (drop / buildUp / verse): {pc.get('drop', 0):.3f} / {pc.get('buildUp', 0):.3f} / {pc.get('verse', 0):.3f}
  - Full per-class: {json.dumps({k: f'{v:.3f}' for k, v in pc.items()})}
  - Verse support per fold: {verse_support}
  - Train time: {m.get('train_time_total', 0):.1f}s  |  Infer time: {m.get('infer_time_total', 0):.3f}s
- **DECISION:** {decision} — {decision_reason}
- **notes:** {notes}

---
"""
    with open(RESULTS_PATH, "a") as f:
        f.write(block)

    record = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "timestamp": ts,
        "wall_time": wall_time,
        "hypothesis": hypothesis,
        "config": config,
        "metrics": {
            "macro_f1_mean": m.get("macro_f1_mean"),
            "macro_f1_std": m.get("macro_f1_std"),
            "macro_f1_per_fold": fold_f1s,
            "boundary_f1_mean": m.get("boundary_f1_mean"),
            "boundary_f1_std": m.get("boundary_f1_std"),
            "drop_onset_error_median": m.get("drop_onset_error_median"),
            "per_class_mean": pc,
            "verse_support_per_fold": verse_support,
        },
        "decision": decision,
        "decision_reason": decision_reason,
        "notes": notes,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_insight(text: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(INSIGHTS_PATH, "a") as f:
        f.write(f"\n[{ts}] {text}\n")


def log_error(exp_id: str, error: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(ERRORS_PATH, "a") as f:
        f.write(f"\n[{ts}] {exp_id}:\n{error}\n")


def log_blocked(exp_id: str, reason: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(BLOCKED_PATH, "a") as f:
        f.write(f"\n[{ts}] {exp_id}: {reason}\n")


# ---------------------------------------------------------------------------
# State (resumption)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"completed": [], "current_baseline_config": None, "best_config": None,
            "best_macro_f1": -1.0, "best_exp_id": None}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Significance check
# ---------------------------------------------------------------------------

def is_significant_improvement(new_f1s: List[float], baseline_f1s: List[float]) -> bool:
    """Keep only if mean improvement > 1 std of the new scores."""
    if not baseline_f1s:
        return True
    delta = np.mean(new_f1s) - np.mean(baseline_f1s)
    return delta > np.std(new_f1s)


# ---------------------------------------------------------------------------
# Confusion matrix text renderer
# ---------------------------------------------------------------------------

def cm_to_text(cm: list, labels: list) -> str:
    w = max(len(l) for l in labels) + 2
    header = " " * w + "  ".join(f"{l:>{w}}" for l in labels)
    rows = [header]
    for i, row in enumerate(cm):
        rows.append(f"{labels[i]:>{w}}  " + "  ".join(f"{v:>{w}}" for v in row))
    return "\n".join(rows)
