"""
Tuning: (1) XGBoost hyperparameter grid, (2) Viterbi stickiness sweep.
Runs on the winning feature spec (EXP-9v) with frozen folds, so results are directly
comparable to the 0.842 baseline. Logs to results.md / insights.md. Saves best tuned bundle.
"""
import sys, time, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score

import harness as H
from assemble import FeatureSpec, assemble
from features import build_cache
import smoothing as S

SPEC = FeatureSpec(use_logmel=True,
    scalar_cols=["rms_mean","rms_std","onset_density","centroid_mean","zcr_mean"],
    per_track_norm=True, context_bars=8)
CLASSES = H.ALL_CLASSES
RS = 42

GRID = [
    dict(max_depth=6, learning_rate=0.1,  n_estimators=300),   # current default (reference)
    dict(max_depth=4, learning_rate=0.05, n_estimators=600),
    dict(max_depth=6, learning_rate=0.05, n_estimators=600),
    dict(max_depth=8, learning_rate=0.05, n_estimators=400),
    dict(max_depth=6, learning_rate=0.1,  n_estimators=600, min_child_weight=3),
    dict(max_depth=5, learning_rate=0.05, n_estimators=600, reg_lambda=3.0),
]


def aligned_proba(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((raw.shape[0], len(CLASSES)))
    for col, c in enumerate(model.classes_):
        p[:, int(c)] = raw[:, col]
    return p


def boundary_and_macro(val_df, y_true, y_pred):
    valid = [c for c in CLASSES if c in set(y_true)]
    macro = f1_score(y_true, y_pred, average="macro", labels=valid, zero_division=0)
    bfs, derrs = [], []
    for _, tdf in val_df.groupby("track", sort=False):
        if len(tdf) < 2: continue
        rows = tdf.index.to_numpy()
        bfs.append(H.boundary_f1(y_true[rows], y_pred[rows]))
        e = H.drop_onset_error(y_true[rows], y_pred[rows])
        if e is not None: derrs.append(e)
    return macro, float(np.mean(bfs)), (float(np.median(derrs)) if derrs else None)


def main():
    cache = build_cache()
    fold_map = H.build_or_load_folds(sorted(cache["track"].unique().tolist()))
    bar_df = assemble(cache, SPEC)
    le = LabelEncoder().fit(CLASSES)
    fold_ids = sorted(set(fold_map.values()))

    # Pre-split per fold (assemble once)
    folds = []
    for f in fold_ids:
        vt = [t for t, ff in fold_map.items() if ff == f]
        vdf = bar_df[bar_df["track"].isin(vt) & (bar_df["label"] != "")].sort_values(["track","bar_index"]).reset_index(drop=True)
        tdf = bar_df[~bar_df["track"].isin(vt) & (bar_df["label"] != "")].sort_values(["track","bar_index"]).reset_index(drop=True)
        folds.append((tdf, vdf))

    print("=== (1) XGBoost hyperparameter grid ===")
    grid_results = []
    proba_cache = {}  # config_idx -> list of (val_df, proba, y_true)
    for ci, params in enumerate(GRID):
        t0 = time.perf_counter()
        macros, bf1s, derrs = [], [], []
        per_fold_proba = []
        for (tdf, vdf) in folds:
            Xtr = np.stack(tdf["feature"].to_list())
            ytr = le.transform([l if l in set(le.classes_) else le.classes_[0] for l in tdf["label"]])
            Xv = np.stack(vdf["feature"].to_list())
            yv = np.array(vdf["label"].tolist())
            sw = compute_sample_weight("balanced", ytr)
            clf = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=RS,
                                eval_metric="mlogloss", subsample=0.9, colsample_bytree=0.9,
                                **params)
            clf.fit(Xtr, ytr, sample_weight=sw)
            p = aligned_proba(clf, Xv)
            trans = S.learn_transition_matrix(tdf, CLASSES, smoothing=1.0)
            pred_idx = np.empty(len(vdf), dtype=int)
            for _, g in vdf.groupby("track", sort=False):
                r = g.index.to_numpy()
                pred_idx[r] = S.viterbi_decode(p[r], trans)
            yp = le.inverse_transform(pred_idx)
            m, b, d = boundary_and_macro(vdf, yv, yp)
            macros.append(m); bf1s.append(b)
            if d is not None: derrs.append(d)
            per_fold_proba.append((vdf, p, yv, trans))
        proba_cache[ci] = per_fold_proba
        dt = time.perf_counter() - t0
        res = dict(params=params, macro=float(np.mean(macros)), macro_std=float(np.std(macros)),
                   boundary=float(np.mean(bf1s)), drop_err=(float(np.median(derrs)) if derrs else None), time=dt)
        grid_results.append(res)
        print(f"  [{ci}] {params} -> macro {res['macro']:.4f}±{res['macro_std']:.4f} "
              f"boundary {res['boundary']:.4f} drop {res['drop_err']} ({dt:.0f}s)")
        H.log_insight(f"TUNE-xgb[{ci}] {params}: macro={res['macro']:.4f} boundary={res['boundary']:.4f}")

    best_ci = int(np.argmax([r["macro"] for r in grid_results]))
    best = grid_results[best_ci]
    print(f"\nBest XGB config: [{best_ci}] {best['params']} macro={best['macro']:.4f}")

    print("\n=== (2) Viterbi stickiness sweep (on best config, reusing cached proba) ===")
    stick_results = []
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        macros, bf1s, derrs = [], [], []
        for (vdf, p, yv, trans) in proba_cache[best_ci]:
            T = trans + alpha * np.eye(len(CLASSES))
            T = T / T.sum(axis=1, keepdims=True)
            pred_idx = np.empty(len(vdf), dtype=int)
            for _, g in vdf.groupby("track", sort=False):
                r = g.index.to_numpy()
                pred_idx[r] = S.viterbi_decode(p[r], T)
            yp = le.inverse_transform(pred_idx)
            m, b, d = boundary_and_macro(vdf, yv, yp)
            macros.append(m); bf1s.append(b)
            if d is not None: derrs.append(d)
        r = dict(alpha=alpha, macro=float(np.mean(macros)), boundary=float(np.mean(bf1s)),
                 drop_err=(float(np.median(derrs)) if derrs else None))
        stick_results.append(r)
        print(f"  alpha={alpha:<4} macro {r['macro']:.4f} boundary {r['boundary']:.4f} drop {r['drop_err']}")
        H.log_insight(f"TUNE-stick alpha={alpha}: macro={r['macro']:.4f} boundary={r['boundary']:.4f}")

    best_alpha = max(stick_results, key=lambda r: r["boundary"])
    print(f"\nBest stickiness alpha={best_alpha['alpha']} (boundary {best_alpha['boundary']:.4f})")

    # Persist tuning summary
    out = {"grid": grid_results, "best_config": best["params"], "best_macro": best["macro"],
           "stickiness": stick_results, "best_alpha": best_alpha["alpha"]}
    (H.EXPERIMENTS_DIR / "tuning_results.json").write_text(json.dumps(out, indent=2, default=float))

    with open(H.RESULTS_PATH, "a") as f:
        f.write(f"\n## TUNING (XGB grid + Viterbi stickiness)\n")
        f.write(f"- Baseline (EXP-9v): macro 0.842, boundary 0.792\n")
        f.write(f"- Best XGB config: {best['params']} -> macro {best['macro']:.4f}, boundary {best['boundary']:.4f}\n")
        f.write(f"- Best stickiness alpha={best_alpha['alpha']}: macro {best_alpha['macro']:.4f}, boundary {best_alpha['boundary']:.4f}\n")
        for r in grid_results:
            f.write(f"    grid {r['params']}: macro {r['macro']:.4f}±{r['macro_std']:.4f} boundary {r['boundary']:.4f}\n")
        for r in stick_results:
            f.write(f"    stick a={r['alpha']}: macro {r['macro']:.4f} boundary {r['boundary']:.4f}\n")
        f.write("\n---\n")

    print("\nTuning complete. Summary -> experiments/tuning_results.json + results.md")
    print(f"VERDICT: best macro {best['macro']:.4f} (vs 0.842), best boundary {best_alpha['boundary']:.4f} (vs 0.792)")


if __name__ == "__main__":
    main()
