"""
Experiment loop driver. Runs pending experiments from the queue, one variable at a time,
cumulative baseline, significance-gated keep/revert. Resumable via state.json.

Usage: .venv/bin/python3 experiments/run_experiments.py
"""
from __future__ import annotations

import copy
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))

import joblib
import numpy as np

import harness as H
from assemble import FeatureSpec, assemble
from features import build_cache
import models as M
import smoothing as S

# Baseline feature spec (matches EXP-0)
BASELINE_SPEC = FeatureSpec(use_logmel=True, scalar_cols=[], per_track_norm=False,
                            add_delta=False, add_position=False, context_bars=8)

MODEL_FACTORIES = {"mlp": M.make_mlp, "rf": M.make_random_forest, "xgb": M.make_xgboost}


# ---------------------------------------------------------------------------
# Experiment queue. Each is a dict describing ONE change vs cumulative baseline.
# kind: "feature" | "context" | "model" | "smooth"
# ---------------------------------------------------------------------------
QUEUE = [
    dict(id="EXP-1", name="+ RMS/energy", kind="feature",
         hypothesis="Energy is more predictive than timbre (drops/quiet are energy-first).",
         spec_change=dict(add_scalars=["rms_mean", "rms_std"])),
    dict(id="EXP-2", name="+ per-track normalization", kind="feature",
         hypothesis="Normalizing per track removes mastering/loudness confound across tracks.",
         spec_change=dict(per_track_norm=True), modifier=True),
    dict(id="EXP-3", name="+ onset density (kick proxy)", kind="feature",
         hypothesis="Onset/kick density separates buildUp/drop/preDropFill.",
         spec_change=dict(add_scalars=["onset_density"])),
    dict(id="EXP-4", name="+ centroid + ZCR", kind="feature",
         hypothesis="Brightness/noisiness scalars may help lead/verse separation.",
         spec_change=dict(add_scalars=["centroid_mean", "zcr_mean"])),
    dict(id="EXP-5", name="+ position-in-track", kind="feature",
         hypothesis="Phrases have positional tendencies. WATCH overfitting to artist structure.",
         spec_change=dict(add_position=True)),
    dict(id="EXP-6", name="+ delta features", kind="feature",
         hypothesis="buildUp is defined by rising trajectory, not absolute level.",
         spec_change=dict(add_delta=True)),
    dict(id="EXP-7", name="CONTEXT_BARS=2", kind="context",
         hypothesis="Less context may suffice / reduce overfitting with richer features.",
         spec_change=dict(context_bars=2)),
    dict(id="EXP-8", name="Model -> RandomForest", kind="model",
         hypothesis="Trees handle mixed-scale features + the overfitting gap better than MLP.",
         model="rf"),
    dict(id="EXP-9", name="Model -> XGBoost", kind="model",
         hypothesis="Boosted trees may beat RF; CPU-only.",
         model="xgb", conditional="rf_promising"),
    dict(id="EXP-10", name="+ Viterbi/HMM smoothing", kind="smooth",
         hypothesis="Enforcing phrase runs + plausible transitions should lift boundary F1.",
         smooth="viterbi", judge_metric="boundary_f1"),
    dict(id="EXP-11", name="+ median smoothing", kind="smooth",
         hypothesis="Mode filter removes 1-bar flicker; compare vs Viterbi.",
         smooth="median", judge_metric="boundary_f1"),
]


def spec_from_dict(d: dict) -> FeatureSpec:
    return FeatureSpec(**d)


def apply_change(spec: FeatureSpec, change: dict) -> FeatureSpec:
    new = copy.deepcopy(spec)
    if "add_scalars" in change:
        for c in change["add_scalars"]:
            if c not in new.scalar_cols:
                new.scalar_cols.append(c)
    if "per_track_norm" in change:
        new.per_track_norm = change["per_track_norm"]
    if "add_position" in change:
        new.add_position = change["add_position"]
    if "add_delta" in change:
        new.add_delta = change["add_delta"]
    if "context_bars" in change:
        new.context_bars = change["context_bars"]
    return new


def make_viterbi_smoother():
    def train_hmm(train_df, classes):
        return S.learn_transition_matrix(train_df, classes, smoothing=1.0)
    def smoother(proba_track, classes, trans):
        log_emit = np.log(proba_track + 1e-12)
        log_trans = np.log(trans + 1e-12)
        return S.viterbi_decode(log_emit, log_trans)
    return train_hmm, smoother


def make_median_smoother():
    def smoother(proba_track, classes, hmm):
        argmax = np.argmax(proba_track, axis=1)
        sm = S.median_smooth(argmax, k=1)
        return sm
    return None, smoother


def judge(new_f1s, baseline_f1s, new_metrics, baseline_metrics, judge_metric="macro_f1"):
    """Return (keep: bool, reason: str)."""
    if judge_metric == "macro_f1":
        delta = np.mean(new_f1s) - np.mean(baseline_f1s)
        thresh = np.std(new_f1s)
        if delta > thresh:
            return True, f"Macro F1 +{delta:.4f} > 1std ({thresh:.4f}) — KEEP"
        return False, f"Macro F1 {delta:+.4f} <= 1std ({thresh:.4f}) — neutral/revert"
    elif judge_metric == "boundary_f1":
        nb = new_metrics.get("boundary_f1_mean") or 0
        bb = baseline_metrics.get("boundary_f1_mean") or 0
        nstd = new_metrics.get("boundary_f1_std") or 0
        macro_delta = np.mean(new_f1s) - np.mean(baseline_f1s)
        macro_thresh = np.std(new_f1s)
        bd = nb - bb
        # Keep if boundary improves beyond noise AND macro not significantly hurt
        if bd > nstd and macro_delta > -macro_thresh:
            return True, f"Boundary F1 +{bd:.4f} > 1std ({nstd:.4f}), macro not hurt — KEEP"
        return False, f"Boundary F1 {bd:+.4f} (std {nstd:.4f}) / macro {macro_delta:+.4f} — revert"
    return False, "unknown judge metric"


def save_cms(metrics, exp_id):
    for fr in metrics["fold_results"]:
        cm_text = H.cm_to_text(fr["confusion_matrix"], fr["cm_labels"])
        (H.ARTIFACTS_DIR / f"{exp_id.lower()}_fold{fr['fold']}_cm.txt").write_text(cm_text)


def main():
    print("Loading feature cache...")
    cache = build_cache()  # cached, no recompute
    tracks = sorted(cache["track"].unique().tolist())
    fold_map = H.build_or_load_folds(tracks)

    state = H.load_state()
    # Initialise cumulative tracking if absent
    if "current_feature_spec" not in state:
        state["current_feature_spec"] = BASELINE_SPEC.to_dict()
    if "current_model" not in state:
        state["current_model"] = "mlp"
    state.setdefault("rf_promising", False)
    H.save_state(state)

    n_done = 0
    for exp in QUEUE:
        if exp["id"] in state["completed"]:
            continue

        # Conditional experiments
        if exp.get("conditional") == "rf_promising" and not state.get("rf_promising"):
            H.log_blocked(exp["id"], "Skipped: RandomForest was not promising (no XGB run).")
            state["completed"].append(exp["id"])
            H.save_state(state)
            continue

        print(f"\n{'='*60}\nRunning {exp['id']}: {exp['name']}\n{'='*60}")
        cur_spec = spec_from_dict(state["current_feature_spec"])
        cur_model = state["current_model"]
        baseline_f1s = state["current_baseline_f1s"]
        baseline_metrics = state.get("current_baseline_metrics", {})

        wall = time.perf_counter()
        try:
            # Determine candidate spec, model, smoothing
            cand_spec = cur_spec
            cand_model = cur_model
            train_hmm = None
            smoother = None

            if exp["kind"] in ("feature", "context"):
                cand_spec = apply_change(cur_spec, exp["spec_change"])
            elif exp["kind"] == "model":
                cand_model = exp["model"]
            elif exp["kind"] == "smooth":
                if exp["smooth"] == "viterbi":
                    train_hmm, smoother = make_viterbi_smoother()
                elif exp["smooth"] == "median":
                    train_hmm, smoother = make_median_smoother()

            # Model factory (handle xgb unavailable)
            build_fn, model_cfg = MODEL_FACTORIES[cand_model]()
            if build_fn is None:
                H.log_blocked(exp["id"], f"Model '{cand_model}' unavailable (not installed).")
                state["completed"].append(exp["id"])
                H.save_state(state)
                continue

            # Assemble features
            bar_df = assemble(cache, cand_spec)
            fdim = len(bar_df.iloc[0]["feature"])

            metrics = H.run_cv(
                bar_df=bar_df, fold_map=fold_map, build_model_fn=build_fn,
                label_classes=H.ALL_CLASSES, train_hmm_fn=train_hmm, smoother=smoother,
            )
            wall_time = time.perf_counter() - wall

            # Decide
            judge_metric = exp.get("judge_metric", "macro_f1")
            keep, reason = judge(metrics["macro_f1_per_fold"], baseline_f1s,
                                 metrics, baseline_metrics, judge_metric)

            # Thin-class watch
            pc = metrics["per_class_mean"]
            base_pc = baseline_metrics.get("per_class_mean", {})
            thin_flags = []
            for tc in ["preDropFill", "buildUp", "verse"]:
                if base_pc and pc.get(tc, 0) < base_pc.get(tc, 0) - 0.03:
                    thin_flags.append(f"{tc} {base_pc.get(tc,0):.3f}->{pc.get(tc,0):.3f}")
            thin_note = (" THIN-CLASS REGRESSION: " + ", ".join(thin_flags)) if thin_flags else ""

            config = {**cand_spec.to_dict(), **model_cfg, "feature_dim": fdim,
                      "kind": exp["kind"]}
            if exp["kind"] == "smooth":
                config["smoothing"] = exp["smooth"]

            decision = "KEEP" if keep else "REVERT (neutral)"
            H.log_result(
                exp_id=exp["id"], exp_name=exp["name"], config=config, metrics=metrics,
                hypothesis=exp["hypothesis"], decision=decision,
                decision_reason=reason + thin_note,
                notes=f"judge_metric={judge_metric}. {'modifier on top of features. ' if exp.get('modifier') else ''}"
                      f"cumulative_features={cand_spec.to_dict()}",
                wall_time=wall_time,
            )
            save_cms(metrics, exp["id"])

            # Update state
            if keep:
                if exp["kind"] in ("feature", "context"):
                    state["current_feature_spec"] = cand_spec.to_dict()
                elif exp["kind"] == "model":
                    state["current_model"] = cand_model
                    if cand_model == "rf":
                        state["rf_promising"] = True
                state["current_baseline_f1s"] = metrics["macro_f1_per_fold"]
                state["current_baseline_metrics"] = {
                    "boundary_f1_mean": metrics.get("boundary_f1_mean"),
                    "boundary_f1_std": metrics.get("boundary_f1_std"),
                    "per_class_mean": pc,
                }
                if metrics["macro_f1_mean"] > state.get("best_macro_f1", 0):
                    state["best_macro_f1"] = metrics["macro_f1_mean"]
                    state["best_exp_id"] = exp["id"]
                    state["best_config"] = config
                # Save model artifact for current-best (retrain on all data)
                _save_best_model(cache, cand_spec, build_fn, exp["id"])
            else:
                # rf "promising" even if not kept: if macro within noise but trains fast
                if exp["kind"] == "model" and cand_model == "rf":
                    md = np.mean(metrics["macro_f1_per_fold"]) - np.mean(baseline_f1s)
                    if md > -np.std(metrics["macro_f1_per_fold"]):
                        state["rf_promising"] = True

            state["completed"].append(exp["id"])
            H.save_state(state)

            # Console summary
            bf = metrics.get("boundary_f1_mean")
            de = metrics.get("drop_onset_error_median")
            print(f"  Macro F1: {metrics['macro_f1_mean']:.4f} ± {metrics['macro_f1_std']:.4f}"
                  f"  (baseline mean {np.mean(baseline_f1s):.4f})")
            print(f"  Boundary F1: {bf:.4f}" if bf is not None else "  Boundary F1: N/A",
                  f" | Drop-onset err: {de:.1f}b" if de is not None else " | Drop-onset: N/A")
            print(f"  drop/buildUp/preDropFill/verse: "
                  f"{pc.get('drop',0):.3f}/{pc.get('buildUp',0):.3f}/{pc.get('preDropFill',0):.3f}/{pc.get('verse',0):.3f}")
            print(f"  DECISION: {decision} — {reason}{thin_note}")
            H.log_insight(f"{exp['id']} ({exp['name']}): macroF1={metrics['macro_f1_mean']:.4f}, "
                         f"boundaryF1={bf}, decision={decision}. {reason}{thin_note}")

            n_done += 1
            if n_done % 3 == 0:
                _checkpoint(state)

        except Exception:
            tb = traceback.format_exc()
            H.log_error(exp["id"], tb)
            print(f"  ERROR in {exp['id']} — logged, continuing:\n{tb}")
            # mark failed but completed so loop doesn't retry endlessly
            state.setdefault("failed", []).append(exp["id"])
            state["completed"].append(exp["id"])
            H.save_state(state)

    _checkpoint(state)
    print("\nAll queued experiments processed. Run summarize.py for the final report.")


def _save_best_model(cache, spec, build_fn, exp_id):
    from sklearn.preprocessing import LabelEncoder
    bar_df = assemble(cache, spec)
    bar_df = bar_df[bar_df["label"] != ""]
    le = LabelEncoder().fit(H.ALL_CLASSES)
    X = np.stack(bar_df["feature"].to_list())
    y = le.transform([l if l in set(le.classes_) else le.classes_[0] for l in bar_df["label"]])
    model = build_fn(X, y)
    joblib.dump({"model": model, "label_encoder": le, "spec": spec.to_dict()},
                H.ARTIFACTS_DIR / "current_best.joblib")


def _checkpoint(state):
    done = [e for e in state["completed"] if e != "EXP-0"]
    print(f"\n--- CHECKPOINT ---")
    print(f"Completed: {state['completed']}")
    print(f"Current best: {state.get('best_exp_id')} @ Macro F1 {state.get('best_macro_f1', 0):.4f}")
    print(f"Cumulative features: {state.get('current_feature_spec')}")
    print(f"Current model: {state.get('current_model')}")
    print(f"------------------\n")


if __name__ == "__main__":
    main()
