"""
EXP-0: Baseline — current pipeline, k-fold CV, freeze folds, log fresh baseline.
Establishes the evaluation harness and reference numbers. Nothing is changed —
we simply re-evaluate what already exists under the proper eval scheme.
"""
import sys
import time
import traceback
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

import harness as H
from phrase_detection_v1 import (
    CONTEXT_BARS,
    N_MELS,
    SR,
    BEATS_PER_BAR,
    RANDOM_STATE,
    build_bar_dataset,
    LABEL_DIR,
    AUDIO_DIR,
)

EXP_ID = "EXP-0"
EXP_NAME = "Baseline (current pipeline, k-fold)"

ALL_CLASSES = H.ALL_CLASSES  # canonical ordering from harness

# Detect actual config from the copied script
BASELINE_CONFIG = {
    "exp_id": EXP_ID,
    "features": ["log_mel_mean_std"],
    "CONTEXT_BARS": CONTEXT_BARS,
    "N_MELS": N_MELS,
    "SR": SR,
    "BEATS_PER_BAR": BEATS_PER_BAR,
    "hop_length": 512,
    "n_fft": 2048,
    "model": "MLPClassifier",
    "hidden_layers": (256, 128),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 1e-3,
    "max_iter": 200,
    "early_stopping": True,
    "imbalance": "compute_sample_weight(balanced)",
    "RANDOM_STATE": RANDOM_STATE,
    "fold_seed": H.FOLD_SEED,
    "n_folds": H.N_FOLDS,
    "per_track_normalization": False,
}


def build_model(X_train, y_train):
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf


def main():
    state = H.load_state()
    if EXP_ID in state["completed"]:
        print(f"{EXP_ID} already complete — skipping.")
        return state

    print(f"\n{'='*60}")
    print(f"Starting {EXP_ID}: {EXP_NAME}")
    print(f"Detected baseline config: CONTEXT_BARS={CONTEXT_BARS}, N_MELS={N_MELS}, SR={SR}")
    print(f"{'='*60}\n")

    wall_start = time.perf_counter()

    try:
        print("Loading dataset...")
        bar_df = build_bar_dataset(
            label_dir=LABEL_DIR,
            audio_dir=AUDIO_DIR,
        )
        print(f"Loaded {len(bar_df)} bars from {bar_df['track'].nunique()} tracks")
        print(f"Label distribution:\n{bar_df['label'].value_counts()}\n")

        tracks = sorted(bar_df["track"].unique().tolist())
        fold_map = H.build_or_load_folds(tracks)

        # Log fold composition
        from collections import defaultdict
        fold_tracks = defaultdict(list)
        for t, f in fold_map.items():
            fold_tracks[f].append(t)
        H.log_insight(f"{EXP_ID}: fold composition — {dict(fold_tracks)}")

        # Check verse coverage per fold
        for fold, ftracks in fold_tracks.items():
            fold_df = bar_df[bar_df["track"].isin(ftracks)]
            verse_n = (fold_df["label"] == "verse").sum()
            H.log_insight(f"  fold {fold} val: {len(ftracks)} tracks, {verse_n} verse bars")

        print("Running 5-fold CV...")
        metrics = H.run_cv(
            bar_df=bar_df,
            fold_map=fold_map,
            build_model_fn=build_model,
            label_classes=ALL_CLASSES,
        )

        wall_time = time.perf_counter() - wall_start

        # Save confusion matrices per fold
        for fr in metrics["fold_results"]:
            cm_text = H.cm_to_text(fr["confusion_matrix"], fr["cm_labels"])
            cm_path = H.ARTIFACTS_DIR / f"exp0_fold{fr['fold']}_cm.txt"
            cm_path.write_text(cm_text)

        # Save the baseline model from the last fold for reference
        # (not full CV ensemble — just a reference checkpoint)
        # We re-train on all data for artifact saving only
        print("Saving baseline model artifact...")
        le = LabelEncoder().fit(ALL_CLASSES)
        bar_df_clean = bar_df[bar_df["label"] != ""].copy()
        X_all = np.stack(bar_df_clean["feature"].to_list())
        y_all = le.transform([
            l if l in le.classes_ else le.classes_[0]
            for l in bar_df_clean["label"]
        ])
        baseline_model = build_model(X_all, y_all)
        joblib.dump(
            {"model": baseline_model, "label_encoder": le, "config": BASELINE_CONFIG},
            H.ARTIFACTS_DIR / "exp0_baseline.joblib"
        )

        # Determine significance (baseline — always keep, it IS the baseline)
        decision = "KEEP (establishes baseline)"
        decision_reason = "First run — this is the reference point, nothing to compare against."

        H.log_result(
            exp_id=EXP_ID,
            exp_name=EXP_NAME,
            config=BASELINE_CONFIG,
            metrics=metrics,
            hypothesis="Establish fresh baseline under k-fold + boundary-F1 eval. "
                       "Old single-split number (Macro F1=0.54) is not comparable.",
            decision=decision,
            decision_reason=decision_reason,
            notes=(
                f"Actual CONTEXT_BARS detected from copied script: {CONTEXT_BARS}. "
                f"Confusion matrices saved to experiments/artifacts/exp0_fold*_cm.txt. "
                f"This result is the new reference — all subsequent experiments compared against it."
            ),
            wall_time=wall_time,
        )

        # Update state
        state["completed"].append(EXP_ID)
        state["current_baseline_config"] = BASELINE_CONFIG
        state["current_baseline_f1s"] = metrics["macro_f1_per_fold"]
        state["best_macro_f1"] = metrics["macro_f1_mean"]
        state["best_exp_id"] = EXP_ID
        state["best_config"] = BASELINE_CONFIG
        H.save_state(state)

        # Print summary
        print(f"\n{'='*60}")
        print(f"EXP-0 COMPLETE")
        print(f"Macro F1: {metrics['macro_f1_mean']:.4f} ± {metrics['macro_f1_std']:.4f}")
        print(f"Per-fold: {[f'{x:.3f}' for x in metrics['macro_f1_per_fold']]}")
        if metrics.get('boundary_f1_mean') is not None:
            print(f"Boundary F1: {metrics['boundary_f1_mean']:.4f} ± {metrics['boundary_f1_std']:.4f}")
        if metrics.get('drop_onset_error_median') is not None:
            print(f"Drop-onset error: {metrics['drop_onset_error_median']:.1f} bars")
        pc = metrics["per_class_mean"]
        print(f"Key classes — drop: {pc.get('drop',0):.3f}  buildUp: {pc.get('buildUp',0):.3f}  verse: {pc.get('verse',0):.3f}")
        print(f"Verse support per fold: {metrics['verse_support_per_fold']}")
        print(f"Wall time: {wall_time:.1f}s")
        print(f"Results appended to experiments/results.md")
        print(f"{'='*60}\n")

        H.log_insight(
            f"{EXP_ID} baseline established: Macro F1={metrics['macro_f1_mean']:.4f}±{metrics['macro_f1_std']:.4f}, "
            f"boundary F1={metrics.get('boundary_f1_mean')}, "
            f"drop={pc.get('drop',0):.3f}, buildUp={pc.get('buildUp',0):.3f}, verse={pc.get('verse',0):.3f}"
        )

    except Exception:
        tb = traceback.format_exc()
        H.log_error(EXP_ID, tb)
        print(f"ERROR in {EXP_ID}:\n{tb}")
        raise

    return state


if __name__ == "__main__":
    main()
