# Experiment Queue (ordered)

Baseline = EXP-0 (log-mel mean+std, ±8 bar context, MLP). Macro F1 = 0.689 ± 0.034.
Each experiment changes **one variable** vs the current cumulative baseline. A change is
**kept** only if mean Macro F1 improves by > 1 std across folds (else reverted as "neutral").

Feature experiments are cumulative (kept changes carry forward). Context, model, and
smoothing are tested against whatever cumulative feature baseline exists at that point.

| # | Experiment | Variable changed | Why this order |
|---|-----------|------------------|----------------|
| EXP-0 | Baseline | — | Reference (done) |
| EXP-1 | + RMS/energy | add rms_mean, rms_std | Highest expected impact — drops/quiet are energy-first; cheap |
| EXP-2 | + per-track norm | normalize features per track | Modifier; removes cross-track mastering/loudness confound. Test early so later features inherit it |
| EXP-3 | + onset density | add onset_density (kick proxy) | Separates buildUp/drop/preDropFill by rhythmic density |
| EXP-4 | + centroid + ZCR | add centroid_mean, zcr_mean | Cheap timbre scalars; may help lead/verse |
| EXP-5 | + position-in-track | add normalized bar position | Positional priors; WATCH overfitting (single-artist structure) |
| EXP-6 | + delta features | add Δ vs previous bar | Targets buildUp (rising trajectory) |
| EXP-7 | CONTEXT_BARS=2 | context window 8→2 | Standalone test; does less context help/hurt with richer features? |
| EXP-8 | Model → RandomForest | MLP→RF, same features+imbalance | Zero new dep; trees handle mixed-scale features + overfitting better |
| EXP-9 | Model → XGB/LightGBM | (only if RF promising & budget) | Stronger trees; logs as new CPU dep |
| EXP-10 | + Viterbi/HMM smoothing | post-process probs w/ learned transitions | Enforces "phrases are runs"; expected big boundary-F1 gain |
| EXP-11 | + median smooth / 4-bar snap | post-process | Compare vs Viterbi; boundary cleanup |

Budget cap: 12 experiments. Definition of best: Macro F1 primary, boundary F1 +
drop-onset error co-primary (operational). Per-class F1 for drop/buildUp/preDropFill
tracked (note: with 64 tracks the thin class is now **preDropFill** @ 239 bars, not verse).

## Notes / deviations
- **Kick density**: real kick detector has no saved artifact → using librosa onset density
  as proxy (logged in insights.md).
- Per-track normalization normalizes all base feature columns (log-mel + scalars) using each
  track's own median/std — leakage-safe (uses only that track's data).
