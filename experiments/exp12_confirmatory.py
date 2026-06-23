"""
EXP-12 (confirmatory, +1 over nominal queue): test whether the individually-'neutral' but
trending-positive features STACK on the best model (RF), and whether Viterbi's boundary gain
holds on the combined config. Justified because greedy one-at-a-time under-credits features
that each improve sub-threshold — leaving the true 'best combined' untested would fail the
core goal. Budget note: this is experiment #13 total (cap was ~10-12; deliberate, logged).
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
import harness as H
from assemble import FeatureSpec, assemble
from features import build_cache
import models as M
import smoothing as S
from run_experiments import make_viterbi_smoother

cache = build_cache()
fold_map = H.build_or_load_folds(sorted(cache["track"].unique().tolist()))
state = H.load_state()
base_f1s = state["current_baseline_f1s"]  # EXP-8 RF norm-only f1s
base_metrics = state.get("current_baseline_metrics", {})

# Combined feature set: norm + all trending-positive scalars (exclude delta — hurt preDropFill)
combined = FeatureSpec(
    use_logmel=True,
    scalar_cols=["rms_mean", "rms_std", "onset_density", "centroid_mean", "zcr_mean"],
    per_track_norm=True, add_delta=False, add_position=False, context_bars=8,
)

configs = [
    ("EXP-12a", "RF + norm + all scalar features", combined, "rf", False),
    ("EXP-12b", "RF + norm + all scalars + position",
     FeatureSpec(use_logmel=True, scalar_cols=combined.scalar_cols, per_track_norm=True,
                 add_position=True, context_bars=8), "rf", False),
    ("EXP-12c", "RF + norm + all scalars + Viterbi", combined, "rf", True),
]

results = {}
for exp_id, name, spec, model_key, use_viterbi in configs:
    print(f"\n{'='*60}\n{exp_id}: {name}\n{'='*60}")
    t0 = time.perf_counter()
    build_fn, model_cfg = M.MODEL_FACTORIES[model_key]() if hasattr(M, "MODEL_FACTORIES") else \
        ({"rf": M.make_random_forest}[model_key]())
    train_hmm, smoother = make_viterbi_smoother() if use_viterbi else (None, None)
    bar_df = assemble(cache, spec)
    fdim = len(bar_df.iloc[0]["feature"])
    metrics = H.run_cv(bar_df=bar_df, fold_map=fold_map, build_model_fn=build_fn,
                       label_classes=H.ALL_CLASSES, train_hmm_fn=train_hmm, smoother=smoother)
    wall = time.perf_counter() - t0
    pc = metrics["per_class_mean"]
    md = metrics["macro_f1_mean"] - np.mean(base_f1s)
    bf = metrics.get("boundary_f1_mean")
    config = {**spec.to_dict(), **model_cfg, "feature_dim": fdim,
              "viterbi": use_viterbi, "kind": "confirmatory"}
    keep_macro = md > np.std(metrics["macro_f1_per_fold"])
    reason = f"Macro F1 {md:+.4f} vs EXP-8 (RF norm-only); {'beats' if keep_macro else 'within'} 1std"
    H.log_result(exp_id=exp_id, exp_name=name, config=config, metrics=metrics,
                 hypothesis="Do trending-positive features stack on RF? Does Viterbi hold?",
                 decision="CANDIDATE" if keep_macro or (bf and bf > (base_metrics.get('boundary_f1_mean') or 0)) else "neutral",
                 decision_reason=reason, notes="confirmatory combined test", wall_time=wall)
    H.save_state(state)  # no cumulative change; confirmatory
    for fr in metrics["fold_results"]:
        (H.ARTIFACTS_DIR / f"{exp_id.lower()}_fold{fr['fold']}_cm.txt").write_text(
            H.cm_to_text(fr["confusion_matrix"], fr["cm_labels"]))
    print(f"  Macro F1: {metrics['macro_f1_mean']:.4f} ± {metrics['macro_f1_std']:.4f} (EXP-8: {np.mean(base_f1s):.4f})")
    print(f"  Boundary F1: {bf:.4f} (EXP-8: {base_metrics.get('boundary_f1_mean'):.4f}) | drop-onset: {metrics.get('drop_onset_error_median')}b")
    print(f"  drop/buildUp/preDropFill/verse: {pc.get('drop',0):.3f}/{pc.get('buildUp',0):.3f}/{pc.get('preDropFill',0):.3f}/{pc.get('verse',0):.3f}")
    results[exp_id] = (metrics, spec, build_fn)
    H.log_insight(f"{exp_id} ({name}): macroF1={metrics['macro_f1_mean']:.4f}, boundaryF1={bf}. {reason}")

# Persist best combined model + update state if it beats current best
best_id = max(results, key=lambda k: results[k][0]["macro_f1_mean"])
best_metrics, best_spec, best_build = results[best_id]
if best_metrics["macro_f1_mean"] > state.get("best_macro_f1", 0):
    print(f"\n{best_id} is new overall best (Macro F1 {best_metrics['macro_f1_mean']:.4f})")
    state["best_macro_f1"] = best_metrics["macro_f1_mean"]
    state["best_exp_id"] = best_id
    state["best_config"] = {**best_spec.to_dict(), "model": "rf",
                            "viterbi": "12c" in best_id}
    from sklearn.preprocessing import LabelEncoder
    bdf = assemble(cache, best_spec); bdf = bdf[bdf["label"] != ""]
    le = LabelEncoder().fit(H.ALL_CLASSES)
    X = np.stack(bdf["feature"].to_list())
    y = le.transform([l if l in set(le.classes_) else le.classes_[0] for l in bdf["label"]])
    joblib.dump({"model": best_build(X, y), "label_encoder": le, "spec": best_spec.to_dict()},
                H.ARTIFACTS_DIR / "current_best.joblib")
for e in ["EXP-12a", "EXP-12b", "EXP-12c"]:
    if e not in state["completed"]:
        state["completed"].append(e)
H.save_state(state)
print("\nConfirmatory experiments done.")
