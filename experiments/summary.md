# Phrase Detection — Experiment Summary

**Run date:** 2026-06-23  |  **Dataset:** 64 tracks, 8009 bars, 5-fold track-level CV (frozen)
**Eval:** Macro F1 (mean±std across folds) primary; Boundary F1 (±1 bar) + drop-onset error co-primary.

---

## TL;DR — Recommendation

**Use: XGBoost + per-track-normalized log-mel + scalar features (RMS, onset-density,
spectral-centroid, ZCR) + Viterbi smoothing.** (EXP-9v)

- **Macro F1: 0.842 ± 0.030** (vs 0.689 baseline — **+0.153**, a 22% relative gain)
- **Boundary F1: 0.792** (vs 0.520 baseline — **+0.272**, the biggest operational win)
- **Drop-onset error: 0 bars** (vs 5 bars baseline)
- `preDropFill` (the fragile thin class) at **0.842** — its best result anywhere.
- Saved ready-to-use: `experiments/artifacts/recommended_model.joblib` (model + transition matrix).

XGBoost beat RandomForest on every metric once tested on the winning feature set; Viterbi
adds a further boundary-F1 lift (0.741→0.792) at no Macro-F1 cost. The only trade-off is
training time (~190s vs RF's ~15s) — irrelevant for offline pre-processing. RF + Viterbi
(EXP-12c, Macro F1 0.815 / boundary 0.746) remains the best **dependency-free** option if you
want to avoid the xgboost dependency.

---

## Results table

| Exp | Config (change) | Macro F1 | Boundary F1 | Drop-onset | Verdict |
|-----|-----------------|----------|-------------|------------|---------|
| EXP-0 | Baseline: log-mel, ±8 ctx, MLP | 0.689 ± 0.034 | 0.520 | 5 b | baseline |
| EXP-1 | + RMS/energy | 0.696 ± 0.032 | 0.558 | 3 b | neutral (sub-σ) |
| **EXP-2** | **+ per-track norm** | **0.722 ± 0.026** | 0.559 | 2 b | **KEEP** |
| EXP-3 | + onset density | 0.737 ± 0.027 | 0.608 | 3 b | neutral (sub-σ) |
| EXP-4 | + centroid + ZCR | 0.750 ± 0.035 | 0.610 | 0 b | neutral (sub-σ) |
| EXP-5 | + position-in-track | 0.740 ± 0.026 | 0.607 | 4 b | neutral (sub-σ) |
| EXP-6 | + delta features | 0.727 ± 0.034 | 0.530 | 4 b | neutral; hurt preDropFill |
| EXP-7 | CONTEXT_BARS=2 | 0.643 ± 0.013 | 0.524 | 7 b | revert (clearly worse) |
| **EXP-8** | **MLP → RandomForest** | **0.770 ± 0.033** | 0.654 | 0 b | **KEEP** |
| EXP-9 | → XGBoost (on winning features) | 0.834 ± 0.037 | 0.741 | 0 b | candidate (beats RF) |
| **EXP-9v** | **XGBoost + Viterbi** | **0.842 ± 0.030** | **0.792** | 0 b | **RECOMMENDED (best overall)** |
| EXP-10 | + Viterbi (on RF norm-only) | 0.772 ± 0.039 | 0.712 | 0 b | neutral by gate¹ |
| EXP-11 | + median smooth | 0.731 ± 0.030 | 0.664 | 0 b | revert; hurt preDropFill |
| EXP-12a | RF + norm + all scalars | 0.815 ± 0.037 | 0.716 | 0 b | candidate |
| **EXP-12b** | + position | **0.821 ± 0.035** | 0.722 | 0 b | best Macro F1 |
| **EXP-12c** | 12a + Viterbi | 0.815 ± 0.029 | **0.746** | 0 b | **RECOMMENDED** |

¹ EXP-10 was auto-reverted by the strict >1σ gate, but its boundary-F1 gain (+0.058) is
operationally real — the gate was too conservative for boundary metrics (see Caveats).

## Trade-off table

| Approach | Macro F1 | Boundary F1 | Train time | Inference | Complexity | Data sensitivity |
|----------|----------|-------------|-----------|-----------|------------|------------------|
| MLP (baseline) | 0.69 | 0.52 | ~30s/5-fold | fast | low | overfits (big train/val gap) |
| MLP + features | 0.75 | 0.61 | ~30s | fast | low | moderate |
| RandomForest + features | 0.82 | 0.72 | ~15s | fast | medium | robust, low overfit |
| RF + features + Viterbi | 0.82 | 0.75 | ~15s + negligible | fast | medium | robust (best dep-free) |
| XGBoost + features | 0.83 | 0.74 | ~190s/5-fold | fast | medium | robust |
| **XGBoost + features + Viterbi** | **0.84** | **0.79** | ~190s + negligible | fast | medium | **best overall** |
| Sequence models (GRU/TCN) | not run | — | — | — | high | data-hungry (future) |

---

## Key findings & insights

1. **Per-track normalization is the highest-leverage single change** (EXP-2: +0.033, the only
   feature change to pass the gate alone). Removing cross-track mastering/loudness differences
   stops the model learning mix volume as signal. Foundational — everything builds on it.

2. **Features stack; the greedy gate hid it.** RMS, onset-density, centroid+ZCR each improved
   Macro F1 by +0.007…+0.028 but individually fell under the 1σ bar and were reverted. Tested
   **together on RF (EXP-12a) they add +0.045** over RF-norm-only. The one-variable-at-a-time
   protocol systematically under-credits features whose individual effect is real but small.

3. **RandomForest > MLP, decisively** (EXP-8: +0.048). Beyond raw score, RF closed the
   overfitting gap the MLP suffered and trains ~2× faster. Trees handle the mixed-scale
   scalar+log-mel feature vector far better than a 2176-dim dense net on 64 tracks.

4. **Viterbi smoothing is where boundary accuracy is won** (EXP-12c: boundary F1 0.716→0.746,
   and tightened fold variance 0.037→0.029) at essentially zero Macro-F1 cost. This is the
   operationally important result for timing lights/pyro. Median smoothing (EXP-11) hurt — it
   wrecked `preDropFill` (0.67→0.40) because that phrase is short and a mode filter erases it.

5. **`preDropFill` is now the fragile class, not `verse`.** With 64 tracks `verse` is abundant
   (F1 ~0.85). `preDropFill` (239 bars) is the thin one and is sensitive to over-smoothing —
   watch it in any future smoothing/aggregation.

6. **Less context hurts a lot** (EXP-7, ±2 bars: −0.08). Phrase identity genuinely needs wide
   temporal context; ±8 is justified.

7. **Position-in-track is a marginal, risky win.** It adds +0.006 Macro F1 (within noise) but
   slightly *hurts* `preDropFill` (0.785→0.766) and is the feature most likely to overfit to
   one artist's song structures. **Recommend leaving it out** until validated on a second
   artist — its gain isn't worth the generalization risk.

---

## Caveats (read before trusting the ranking)

- **Greedy + strict gate is order-dependent and conservative.** The kept-path (EXP-2→EXP-8)
  under-represents the true best, which is why the confirmatory EXP-12 combined run was needed.
  The recommended config is a strong, validated local optimum — not a proven global one.
- **The >1σ "keep" gate is appropriate for guarding against noise but too strict for boundary
  F1** (high fold variance). EXP-10/EXP-12c show Viterbi's boundary gain is real; judge
  smoothing on boundary F1 + stability, not the Macro-F1 gate.
- **`onset_density` is a librosa-onset proxy, not the real kick detector** (which has no saved
  artifact). The signal helped; a true kick-pattern feature may help more.
- Single artist / ~150 BPM hardstyle — none of this is validated cross-tempo or cross-artist.

---

## Not yet tried (future work)
- **LightGBM** — XGBoost (EXP-9/9v, xgboost 3.3.0 CPU) was run and won; LightGBM could be
  tried as a faster-training alternative to XGB (XGB trains ~190s/5-fold vs RF ~15s).
- **XGBoost hyperparameter tuning** — EXP-9v used untuned defaults (300 trees, depth 6) and
  still won; a small grid (depth, learning_rate, n_estimators) likely squeezes out more.
- **Combined config without position + with Viterbi as the locked baseline**, then a fresh
  one-at-a-time pass — the cumulative baseline shifted late, so a few features deserve a re-test
  on top of the EXP-12 config.
- **Sequence models (GRU / TCN / Transformer)** and **pretrained audio embeddings (MERT)** —
  deliberately out of scope at 64 tracks; revisit at 100+ labelled tracks. The +context
  dependence (finding 6) suggests a sequence model could help once data supports it.
- **True kick-detector feature** once that model is saved as an artifact.

## Artifacts
- `experiments/artifacts/current_best.joblib` — best model (EXP-12b config) + label encoder + spec.
- `experiments/artifacts/exp*_fold*_cm.txt` — per-fold confusion matrices.
- `experiments/results.md` / `log.jsonl` — full per-experiment record.
- `experiments/folds.json` — frozen fold assignment (reuse for any follow-up).
