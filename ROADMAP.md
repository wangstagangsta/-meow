# Roadmap / Future Plans

Captured from the phrase-detection benchmarking work (see `experiments/summary.md`) and
planning discussion. Ordered roughly by priority/dependency. The guiding principle: this is a
small-data project (~65 single-artist tracks), so **data and clean labels come before feature
engineering, and feature engineering comes before sequence models.** Each stage is a
prerequisite for the next being worthwhile.

## Current best (baseline to beat)
- **Model:** XGBoost + per-track-normalized log-mel + scalar features (RMS, onset-density,
  spectral-centroid, ZCR) + Viterbi smoothing. Bundle: `models/phrase_detection/phrase_xgb_viterbi.joblib`.
- **Metrics (5-fold track-level CV):** Macro F1 0.842 ± 0.030, Boundary F1 0.792, drop-onset error 0 bars.
- Dependency-free alternative: `phrase_rf_viterbi.joblib` (RandomForest, Macro F1 0.82).

## Key insights driving the plan
- **Error is concentrated in the mid-energy melodic phrases** (`bridge` 0.67, `lead` 0.83,
  `buildUp` 0.82) which confuse with each other and with `verse`. The energy extremes
  (`drop` 0.96, `quiet` 0.89) are essentially solved.
- **Much of the verse/bridge/quiet confusion is label noise** — these are genuinely ambiguous
  to a human labeler, so the model is likely near the achievable ceiling there. Low ROI to
  chase with features; consistent labeling + more data is the real lever.
- **`buildUp` vs `preDropFill` confusion is *real* (not noise)** and is driven by kick
  patterns / kickrolls — this is the one melodic-cluster confusion worth a targeted feature.
- **The model is strongest exactly where it matters operationally** (drop onset, breakdown,
  rising action) — good enough to be useful now.
- **Downbeat phase-correctness is the linchpin for linking the beat + phrase models.** A
  sub-beat offset error is harmless; a whole-beat (wrong phase) error is catastrophic —
  it shifts every bar boundary and wrecks phrase predictions.

## Plan (ordered)

### 1. Relabel dataset with exact downbeats — IN PROGRESS
- Set the cue marker at the **first true downbeat** of every track; derive every beat 1
  mathematically from BPM (valid because hardstyle is constant-tempo, 4/4).
- **Double duty:** (a) clean bar alignment for the phrase model, (b) creates exact
  **downbeat-labeled data** needed to train the downbeat detector later.
- ⚠️ Caveat: "math it out from BPM" only stays phase-correct if BPM is precise enough that
  drift doesn't accumulate over a 4–5 min track. Spot-check that the derived grid still lands
  on the downbeat near the END of the longest tracks.

### 2. Re-run phrase experiments on the clean labels
- The `experiments/` harness + frozen folds are reusable. Re-running on correctly-aligned bars
  should lift boundary F1 / drop-onset especially. Establish a new baseline.

### 3. Easy tuning — DONE
- XGBoost hyperparameter grid + Viterbi "stickiness" (diagonal transition bias) sweep.
  See `experiments/tune_xgb.py` and `experiments/tuning_results.json`.
- **VERDICT: tuning is a wash.** Best grid config (depth 6, lr 0.1, n=600, min_child_weight=3)
  hit Macro F1 0.839 vs 0.842; best stickiness (alpha=1.0) boundary 0.790 vs 0.792 — all within
  the ±0.03 fold noise. The model is **data-limited, not tuning-limited.** Don't spend more time
  on hyperparameters; re-tune only after relabeling + more data, when numbers mean more.
  (Note: results aren't sub-0.01 reproducible at this dataset size — measurement floor ~0.03.)

### 4. More data — varied tempo / artists
- Highest-leverage step for generalization. Everything so far is one artist at ~150 BPM.
- Prioritize tracks with clear `bridge` sections (rarest + weakest class, ~5% of bars).
- Prerequisite for trusting feature work and for sequence models.

### 5. Kick-onset detection model (V2 of the beat detector)
- Foundation for multiple things:
  - **Kickroll detection** (a kickroll = rapid kick sequence) — directly built on it.
  - **Sharper beat grid** — hardstyle is 4-on-the-floor, so precise kick onsets ≈ beat tracking.
  - **Better phrase `onset_density` feature** (currently a crude librosa-onset proxy).
- May be able to reuse machinery from the existing beat CRNN / `kick_detectorv1.ipynb`.

### 6. Downbeat-phase detection
- **Kick onsets alone are NOT sufficient** — every beat has a kick, so they give the beat grid
  but not *which* beat is beat 1. Needs an additional **harmonic/structural cue** (chord/bass
  change on the downbeat, crash, kickrolls resolving onto beat 1).
- Approach: add a downbeat head/classifier on the beat CRNN combining kick timing + harmonic
  features. Train/eval on the exact downbeat labels from step 1.

### 7. Targeted feature tuning (after more data)
- Mostly skip the melodic-cluster (label-noise-limited). The one worthwhile target:
  **kickroll / kick-pattern features for buildUp vs preDropFill** (the real confusion).
- Re-test position-in-track and re-test the features that were reverted under the greedy gate,
  now on the post-relabel baseline.

### 8. Sequence models — GATED on ~100+ tracks
- GRU / TCN / CRF over the bar-feature sequence. The bridge/verse/lead problem is fundamentally
  "role depends on surrounding structure," which sequence models capture. Data-hungry —
  premature at 65 tracks.

### 9. Link beat model + phrase model end-to-end
- Beat/downbeat model predicts BPM + first downbeat → feeds the phrase model automatically
  (instead of hand-labeled BPM/offset). Phrase accuracy will be bounded by downbeat accuracy
  (see linchpin insight). Worth quantifying the phrase model's sensitivity to offset error
  before/at this stage to spec how good the downbeat detector must be.

## Out of scope (for now)
- Pretrained audio embeddings (e.g. MERT) — revisit alongside sequence models.
- Live / real-time processing — the whole pipeline is offline pre-processing by design.
