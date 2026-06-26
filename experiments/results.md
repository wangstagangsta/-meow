
## EXP-0: Baseline (current pipeline, k-fold)
- **timestamp:** 2026-06-23 02:18  |  **wall time:** 33.5s
- **hypothesis:** Establish fresh baseline under k-fold + boundary-F1 eval. Old single-split number (Macro F1=0.54) is not comparable.
- **config:**
```
{
  "exp_id": "EXP-0",
  "features": [
    "log_mel_mean_std"
  ],
  "CONTEXT_BARS": 8,
  "N_MELS": 64,
  "SR": 22050,
  "BEATS_PER_BAR": 4,
  "hop_length": 512,
  "n_fft": 2048,
  "model": "MLPClassifier",
  "hidden_layers": [
    256,
    128
  ],
  "activation": "relu",
  "solver": "adam",
  "learning_rate_init": 0.001,
  "max_iter": 200,
  "early_stopping": true,
  "imbalance": "compute_sample_weight(balanced)",
  "RANDOM_STATE": 42,
  "fold_seed": 2024,
  "n_folds": 5,
  "per_track_normalization": false
}
```
- **metrics:**
  - Macro F1: 0.6891 ± 0.0343
  - Per-fold F1: ['0.641', '0.715', '0.654', '0.723', '0.712']
  - Boundary F1: 0.5198 ± 0.0283
  - Drop-onset error (bars): 5.0
  - Per-class F1 (drop / buildUp / verse): 0.823 / 0.633 / 0.793
  - Full per-class: {"bridge": "0.560", "buildUp": "0.633", "drop": "0.823", "lead": "0.582", "preDropFill": "0.621", "quiet": "0.811", "verse": "0.793"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 13.7s  |  Infer time: 0.008s
- **DECISION:** KEEP (establishes baseline) — First run — this is the reference point, nothing to compare against.
- **notes:** Actual CONTEXT_BARS detected from copied script: 8. Confusion matrices saved to experiments/artifacts/exp0_fold*_cm.txt. This result is the new reference — all subsequent experiments compared against it.

---

## EXP-1: + RMS/energy
- **timestamp:** 2026-06-23 02:32  |  **wall time:** 12.9s
- **hypothesis:** Energy is more predictive than timbre (drops/quiet are energy-first).
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [
    "rms_mean",
    "rms_std"
  ],
  "per_track_norm": false,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "MLPClassifier",
  "hidden_layers": [
    256,
    128
  ],
  "max_iter": 200,
  "early_stopping": true,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 2210,
  "kind": "feature"
}
```
- **metrics:**
  - Macro F1: 0.6958 ± 0.0319
  - Per-fold F1: ['0.666', '0.701', '0.653', '0.735', '0.724']
  - Boundary F1: 0.5575 ± 0.0253
  - Drop-onset error (bars): 3.0
  - Per-class F1 (drop / buildUp / verse): 0.829 / 0.622 / 0.796
  - Full per-class: {"bridge": "0.574", "buildUp": "0.622", "drop": "0.829", "lead": "0.576", "preDropFill": "0.644", "quiet": "0.828", "verse": "0.796"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 12.7s  |  Infer time: 0.011s
- **DECISION:** REVERT (neutral) — Macro F1 +0.0066 <= 1std (0.0319) — neutral/revert
- **notes:** judge_metric=macro_f1. cumulative_features={'use_logmel': True, 'scalar_cols': ['rms_mean', 'rms_std'], 'per_track_norm': False, 'add_delta': False, 'add_position': False, 'context_bars': 8}

---

## EXP-2: + per-track normalization
- **timestamp:** 2026-06-23 02:32  |  **wall time:** 5.8s
- **hypothesis:** Normalizing per track removes mastering/loudness confound across tracks.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "MLPClassifier",
  "hidden_layers": [
    256,
    128
  ],
  "max_iter": 200,
  "early_stopping": true,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 2176,
  "kind": "feature"
}
```
- **metrics:**
  - Macro F1: 0.7221 ± 0.0260
  - Per-fold F1: ['0.708', '0.699', '0.718', '0.713', '0.773']
  - Boundary F1: 0.5590 ± 0.0295
  - Drop-onset error (bars): 2.0
  - Per-class F1 (drop / buildUp / verse): 0.865 / 0.677 / 0.830
  - Full per-class: {"bridge": "0.502", "buildUp": "0.677", "drop": "0.865", "lead": "0.703", "preDropFill": "0.686", "quiet": "0.791", "verse": "0.830"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 5.6s  |  Infer time: 0.008s
- **DECISION:** KEEP — Macro F1 +0.0330 > 1std (0.0260) — KEEP
- **notes:** judge_metric=macro_f1. modifier on top of features. cumulative_features={'use_logmel': True, 'scalar_cols': [], 'per_track_norm': True, 'add_delta': False, 'add_position': False, 'context_bars': 8}

---

## EXP-3: + onset density (kick proxy)
- **timestamp:** 2026-06-23 02:32  |  **wall time:** 6.4s
- **hypothesis:** Onset/kick density separates buildUp/drop/preDropFill.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [
    "onset_density"
  ],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "MLPClassifier",
  "hidden_layers": [
    256,
    128
  ],
  "max_iter": 200,
  "early_stopping": true,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 2193,
  "kind": "feature"
}
```
- **metrics:**
  - Macro F1: 0.7374 ± 0.0271
  - Per-fold F1: ['0.723', '0.732', '0.731', '0.712', '0.790']
  - Boundary F1: 0.6078 ± 0.0477
  - Drop-onset error (bars): 3.0
  - Per-class F1 (drop / buildUp / verse): 0.874 / 0.689 / 0.850
  - Full per-class: {"bridge": "0.536", "buildUp": "0.689", "drop": "0.874", "lead": "0.722", "preDropFill": "0.697", "quiet": "0.794", "verse": "0.850"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 6.2s  |  Infer time: 0.008s
- **DECISION:** REVERT (neutral) — Macro F1 +0.0152 <= 1std (0.0271) — neutral/revert
- **notes:** judge_metric=macro_f1. cumulative_features={'use_logmel': True, 'scalar_cols': ['onset_density'], 'per_track_norm': True, 'add_delta': False, 'add_position': False, 'context_bars': 8}

---

## EXP-4: + centroid + ZCR
- **timestamp:** 2026-06-23 02:32  |  **wall time:** 6.4s
- **hypothesis:** Brightness/noisiness scalars may help lead/verse separation.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [
    "centroid_mean",
    "zcr_mean"
  ],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "MLPClassifier",
  "hidden_layers": [
    256,
    128
  ],
  "max_iter": 200,
  "early_stopping": true,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 2210,
  "kind": "feature"
}
```
- **metrics:**
  - Macro F1: 0.7496 ± 0.0345
  - Per-fold F1: ['0.725', '0.710', '0.760', '0.744', '0.810']
  - Boundary F1: 0.6097 ± 0.0573
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.903 / 0.725 / 0.844
  - Full per-class: {"bridge": "0.519", "buildUp": "0.725", "drop": "0.903", "lead": "0.766", "preDropFill": "0.696", "quiet": "0.795", "verse": "0.844"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 6.2s  |  Infer time: 0.009s
- **DECISION:** REVERT (neutral) — Macro F1 +0.0275 <= 1std (0.0345) — neutral/revert
- **notes:** judge_metric=macro_f1. cumulative_features={'use_logmel': True, 'scalar_cols': ['centroid_mean', 'zcr_mean'], 'per_track_norm': True, 'add_delta': False, 'add_position': False, 'context_bars': 8}

---

## EXP-5: + position-in-track
- **timestamp:** 2026-06-23 02:32  |  **wall time:** 9.0s
- **hypothesis:** Phrases have positional tendencies. WATCH overfitting to artist structure.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": true,
  "context_bars": 8,
  "model": "MLPClassifier",
  "hidden_layers": [
    256,
    128
  ],
  "max_iter": 200,
  "early_stopping": true,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 2193,
  "kind": "feature"
}
```
- **metrics:**
  - Macro F1: 0.7401 ± 0.0258
  - Per-fold F1: ['0.711', '0.728', '0.743', '0.730', '0.787']
  - Boundary F1: 0.6069 ± 0.0460
  - Drop-onset error (bars): 4.0
  - Per-class F1 (drop / buildUp / verse): 0.877 / 0.698 / 0.840
  - Full per-class: {"bridge": "0.546", "buildUp": "0.698", "drop": "0.877", "lead": "0.721", "preDropFill": "0.699", "quiet": "0.799", "verse": "0.840"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 8.8s  |  Infer time: 0.010s
- **DECISION:** REVERT (neutral) — Macro F1 +0.0180 <= 1std (0.0258) — neutral/revert
- **notes:** judge_metric=macro_f1. cumulative_features={'use_logmel': True, 'scalar_cols': [], 'per_track_norm': True, 'add_delta': False, 'add_position': True, 'context_bars': 8}

---

## EXP-6: + delta features
- **timestamp:** 2026-06-23 02:33  |  **wall time:** 10.3s
- **hypothesis:** buildUp is defined by rising trajectory, not absolute level.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [],
  "per_track_norm": true,
  "add_delta": true,
  "add_position": false,
  "context_bars": 8,
  "model": "MLPClassifier",
  "hidden_layers": [
    256,
    128
  ],
  "max_iter": 200,
  "early_stopping": true,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 4352,
  "kind": "feature"
}
```
- **metrics:**
  - Macro F1: 0.7270 ± 0.0344
  - Per-fold F1: ['0.686', '0.702', '0.728', '0.732', '0.787']
  - Boundary F1: 0.5298 ± 0.0606
  - Drop-onset error (bars): 4.0
  - Per-class F1 (drop / buildUp / verse): 0.854 / 0.677 / 0.833
  - Full per-class: {"bridge": "0.586", "buildUp": "0.677", "drop": "0.854", "lead": "0.682", "preDropFill": "0.651", "quiet": "0.807", "verse": "0.833"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 10.0s  |  Infer time: 0.021s
- **DECISION:** REVERT (neutral) — Macro F1 +0.0049 <= 1std (0.0344) — neutral/revert THIN-CLASS REGRESSION: preDropFill 0.686->0.651
- **notes:** judge_metric=macro_f1. cumulative_features={'use_logmel': True, 'scalar_cols': [], 'per_track_norm': True, 'add_delta': True, 'add_position': False, 'context_bars': 8}

---

## EXP-7: CONTEXT_BARS=2
- **timestamp:** 2026-06-23 02:33  |  **wall time:** 3.4s
- **hypothesis:** Less context may suffice / reduce overfitting with richer features.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 2,
  "model": "MLPClassifier",
  "hidden_layers": [
    256,
    128
  ],
  "max_iter": 200,
  "early_stopping": true,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 640,
  "kind": "context"
}
```
- **metrics:**
  - Macro F1: 0.6426 ± 0.0135
  - Per-fold F1: ['0.652', '0.637', '0.642', '0.622', '0.661']
  - Boundary F1: 0.5238 ± 0.0269
  - Drop-onset error (bars): 7.0
  - Per-class F1 (drop / buildUp / verse): 0.836 / 0.604 / 0.788
  - Full per-class: {"bridge": "0.352", "buildUp": "0.604", "drop": "0.836", "lead": "0.608", "preDropFill": "0.592", "quiet": "0.720", "verse": "0.788"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 3.2s  |  Infer time: 0.004s
- **DECISION:** REVERT (neutral) — Macro F1 -0.0795 <= 1std (0.0135) — neutral/revert THIN-CLASS REGRESSION: preDropFill 0.686->0.592, buildUp 0.677->0.604, verse 0.830->0.788
- **notes:** judge_metric=macro_f1. cumulative_features={'use_logmel': True, 'scalar_cols': [], 'per_track_norm': True, 'add_delta': False, 'add_position': False, 'context_bars': 2}

---

## EXP-8: Model -> RandomForest
- **timestamp:** 2026-06-23 02:33  |  **wall time:** 12.8s
- **hypothesis:** Trees handle mixed-scale features + the overfitting gap better than MLP.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "RandomForestClassifier",
  "n_estimators": 300,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
  "feature_dim": 2176,
  "kind": "model"
}
```
- **metrics:**
  - Macro F1: 0.7700 ± 0.0335
  - Per-fold F1: ['0.734', '0.732', '0.776', '0.787', '0.821']
  - Boundary F1: 0.6543 ± 0.0502
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.885 / 0.755 / 0.829
  - Full per-class: {"bridge": "0.642", "buildUp": "0.755", "drop": "0.885", "lead": "0.744", "preDropFill": "0.669", "quiet": "0.867", "verse": "0.829"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 12.4s  |  Infer time: 0.134s
- **DECISION:** KEEP — Macro F1 +0.0479 > 1std (0.0335) — KEEP
- **notes:** judge_metric=macro_f1. cumulative_features={'use_logmel': True, 'scalar_cols': [], 'per_track_norm': True, 'add_delta': False, 'add_position': False, 'context_bars': 8}

---

## EXP-10: + Viterbi/HMM smoothing
- **timestamp:** 2026-06-23 02:33  |  **wall time:** 12.8s
- **hypothesis:** Enforcing phrase runs + plausible transitions should lift boundary F1.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "RandomForestClassifier",
  "n_estimators": 300,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
  "feature_dim": 2176,
  "kind": "smooth",
  "smoothing": "viterbi"
}
```
- **metrics:**
  - Macro F1: 0.7716 ± 0.0386
  - Per-fold F1: ['0.736', '0.766', '0.724', '0.812', '0.819']
  - Boundary F1: 0.7123 ± 0.0656
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.897 / 0.760 / 0.844
  - Full per-class: {"bridge": "0.629", "buildUp": "0.760", "drop": "0.897", "lead": "0.745", "preDropFill": "0.640", "quiet": "0.886", "verse": "0.844"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 12.4s  |  Infer time: 0.134s
- **DECISION:** REVERT (neutral) — Boundary F1 +0.0579 (std 0.0656) / macro +0.0015 — revert
- **notes:** judge_metric=boundary_f1. cumulative_features={'use_logmel': True, 'scalar_cols': [], 'per_track_norm': True, 'add_delta': False, 'add_position': False, 'context_bars': 8}

---

## EXP-11: + median smoothing
- **timestamp:** 2026-06-23 02:33  |  **wall time:** 13.0s
- **hypothesis:** Mode filter removes 1-bar flicker; compare vs Viterbi.
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "RandomForestClassifier",
  "n_estimators": 300,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
  "feature_dim": 2176,
  "kind": "smooth",
  "smoothing": "median"
}
```
- **metrics:**
  - Macro F1: 0.7310 ± 0.0302
  - Per-fold F1: ['0.696', '0.697', '0.747', '0.742', '0.773']
  - Boundary F1: 0.6645 ± 0.0388
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.890 / 0.731 / 0.832
  - Full per-class: {"bridge": "0.640", "buildUp": "0.731", "drop": "0.890", "lead": "0.752", "preDropFill": "0.403", "quiet": "0.868", "verse": "0.832"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 12.6s  |  Infer time: 0.132s
- **DECISION:** REVERT (neutral) — Boundary F1 +0.0101 (std 0.0388) / macro -0.0390 — revert THIN-CLASS REGRESSION: preDropFill 0.669->0.403
- **notes:** judge_metric=boundary_f1. cumulative_features={'use_logmel': True, 'scalar_cols': [], 'per_track_norm': True, 'add_delta': False, 'add_position': False, 'context_bars': 8}

---

## EXP-12a: RF + norm + all scalar features
- **timestamp:** 2026-06-23 02:42  |  **wall time:** 13.4s
- **hypothesis:** Do trending-positive features stack on RF? Does Viterbi hold?
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [
    "rms_mean",
    "rms_std",
    "onset_density",
    "centroid_mean",
    "zcr_mean"
  ],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "RandomForestClassifier",
  "n_estimators": 300,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
  "feature_dim": 2261,
  "viterbi": false,
  "kind": "confirmatory"
}
```
- **metrics:**
  - Macro F1: 0.8145 ± 0.0371
  - Per-fold F1: ['0.752', '0.794', '0.853', '0.829', '0.844']
  - Boundary F1: 0.7162 ± 0.0459
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.947 / 0.800 / 0.846
  - Full per-class: {"bridge": "0.654", "buildUp": "0.800", "drop": "0.947", "lead": "0.801", "preDropFill": "0.785", "quiet": "0.869", "verse": "0.846"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 13.1s  |  Infer time: 0.138s
- **DECISION:** CANDIDATE — Macro F1 +0.0445 vs EXP-8 (RF norm-only); beats 1std
- **notes:** confirmatory combined test

---

## EXP-12b: RF + norm + all scalars + position
- **timestamp:** 2026-06-23 02:42  |  **wall time:** 13.7s
- **hypothesis:** Do trending-positive features stack on RF? Does Viterbi hold?
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [
    "rms_mean",
    "rms_std",
    "onset_density",
    "centroid_mean",
    "zcr_mean"
  ],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": true,
  "context_bars": 8,
  "model": "RandomForestClassifier",
  "n_estimators": 300,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
  "feature_dim": 2278,
  "viterbi": false,
  "kind": "confirmatory"
}
```
- **metrics:**
  - Macro F1: 0.8210 ± 0.0353
  - Per-fold F1: ['0.768', '0.790', '0.843', '0.844', '0.859']
  - Boundary F1: 0.7224 ± 0.0628
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.946 / 0.802 / 0.847
  - Full per-class: {"bridge": "0.687", "buildUp": "0.802", "drop": "0.946", "lead": "0.801", "preDropFill": "0.766", "quiet": "0.898", "verse": "0.847"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 13.3s  |  Infer time: 0.155s
- **DECISION:** CANDIDATE — Macro F1 +0.0509 vs EXP-8 (RF norm-only); beats 1std
- **notes:** confirmatory combined test

---

## EXP-12c: RF + norm + all scalars + Viterbi
- **timestamp:** 2026-06-23 02:43  |  **wall time:** 13.4s
- **hypothesis:** Do trending-positive features stack on RF? Does Viterbi hold?
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [
    "rms_mean",
    "rms_std",
    "onset_density",
    "centroid_mean",
    "zcr_mean"
  ],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "RandomForestClassifier",
  "n_estimators": 300,
  "min_samples_leaf": 2,
  "class_weight": "balanced",
  "feature_dim": 2261,
  "viterbi": true,
  "kind": "confirmatory"
}
```
- **metrics:**
  - Macro F1: 0.8151 ± 0.0290
  - Per-fold F1: ['0.760', '0.819', '0.841', '0.821', '0.836']
  - Boundary F1: 0.7457 ± 0.0521
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.947 / 0.807 / 0.856
  - Full per-class: {"bridge": "0.645", "buildUp": "0.807", "drop": "0.947", "lead": "0.814", "preDropFill": "0.748", "quiet": "0.888", "verse": "0.856"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 12.9s  |  Infer time: 0.157s
- **DECISION:** CANDIDATE — Macro F1 +0.0451 vs EXP-8 (RF norm-only); beats 1std
- **notes:** confirmatory combined test

---

## EXP-9: XGBoost + norm + all scalars
- **timestamp:** 2026-06-23 03:12  |  **wall time:** 192.4s
- **hypothesis:** Do boosted trees beat RF on the winning feature set?
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [
    "rms_mean",
    "rms_std",
    "onset_density",
    "centroid_mean",
    "zcr_mean"
  ],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "XGBClassifier",
  "n_estimators": 300,
  "max_depth": 6,
  "learning_rate": 0.1,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 2261,
  "viterbi": false,
  "kind": "model"
}
```
- **metrics:**
  - Macro F1: 0.8343 ± 0.0371
  - Per-fold F1: ['0.769', '0.818', '0.871', '0.850', '0.863']
  - Boundary F1: 0.7405 ± 0.0648
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.955 / 0.813 / 0.870
  - Full per-class: {"bridge": "0.663", "buildUp": "0.813", "drop": "0.955", "lead": "0.825", "preDropFill": "0.826", "quiet": "0.888", "verse": "0.870"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 192.2s  |  Infer time: 0.035s
- **DECISION:** CANDIDATE — XGB macro 0.8343 vs RF 0.8145; boundary 0.7405 vs RF 0.7162
- **notes:** EXP-9 re-run on winning features (xgb 3.3.0 CPU)

---

## EXP-9v: XGBoost + norm + all scalars + Viterbi
- **timestamp:** 2026-06-23 03:16  |  **wall time:** 191.4s
- **hypothesis:** Do boosted trees beat RF on the winning feature set?
- **config:**
```
{
  "use_logmel": true,
  "scalar_cols": [
    "rms_mean",
    "rms_std",
    "onset_density",
    "centroid_mean",
    "zcr_mean"
  ],
  "per_track_norm": true,
  "add_delta": false,
  "add_position": false,
  "context_bars": 8,
  "model": "XGBClassifier",
  "n_estimators": 300,
  "max_depth": 6,
  "learning_rate": 0.1,
  "imbalance": "balanced_sample_weight",
  "feature_dim": 2261,
  "viterbi": true,
  "kind": "model"
}
```
- **metrics:**
  - Macro F1: 0.8418 ± 0.0296
  - Per-fold F1: ['0.787', '0.834', '0.869', '0.856', '0.862']
  - Boundary F1: 0.7923 ± 0.0628
  - Drop-onset error (bars): 0.0
  - Per-class F1 (drop / buildUp / verse): 0.961 / 0.817 / 0.873
  - Full per-class: {"bridge": "0.673", "buildUp": "0.817", "drop": "0.961", "lead": "0.834", "preDropFill": "0.842", "quiet": "0.892", "verse": "0.873"}
  - Verse support per fold: [469, 603, 511, 439, 427]
  - Train time: 191.2s  |  Infer time: 0.025s
- **DECISION:** CANDIDATE — XGB macro 0.8418 vs RF 0.8151; boundary 0.7923 vs RF 0.7457
- **notes:** EXP-9 re-run on winning features (xgb 3.3.0 CPU)

---

## TUNING (XGB grid + Viterbi stickiness)
- Baseline (EXP-9v): macro 0.842, boundary 0.792
- Best XGB config: {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 600, 'min_child_weight': 3} -> macro 0.8391, boundary 0.7859
- Best stickiness alpha=1.0: macro 0.8401, boundary 0.7898
    grid {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 300}: macro 0.8360±0.0338 boundary 0.7887
    grid {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 600}: macro 0.8332±0.0362 boundary 0.7827
    grid {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 600}: macro 0.8338±0.0304 boundary 0.7863
    grid {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 400}: macro 0.8341±0.0351 boundary 0.7844
    grid {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 600, 'min_child_weight': 3}: macro 0.8391±0.0308 boundary 0.7859
    grid {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 600, 'reg_lambda': 3.0}: macro 0.8337±0.0305 boundary 0.7763
    stick a=0.0: macro 0.8391 boundary 0.7859
    stick a=0.5: macro 0.8391 boundary 0.7858
    stick a=1.0: macro 0.8401 boundary 0.7898
    stick a=2.0: macro 0.8392 boundary 0.7871
    stick a=5.0: macro 0.8381 boundary 0.7859
    stick a=10.0: macro 0.8373 boundary 0.7832

---
