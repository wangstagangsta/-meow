# Task: Explore & Benchmark Phrase Detection Models (Autonomous Single Run)

## Mission
Find the **best model + feature combination** for classifying each bar of a hardstyle
track into a phrase label (`quiet`, `verse`, `lead`, `buildUp`, `preDropFill`, `drop`,
`bridge`), and produce **evidence** for the choice — not just a winning number.

This plan is written to be executed in one long, possibly **unattended/auto** run. It must
be safe to leave running: it self-logs, checkpoints, resumes, and stays inside a sandbox.

---

## ⛔ REPO SAFETY CONTRACT — read first, applies to the entire run

These are hard rules. If any action would violate one, **do not do it — log it to
`experiments/blocked.md` and continue** with the next experiment (or stop if it's fatal).

1. **Writable area = `experiments/` only.** You may create/modify/delete files **only**
   under `experiments/`. Everything else in the repo is **READ-ONLY**.
2. **Never modify in place:** `scripts/`, `notebooks/`, `models/`, `data/`, `README.md`,
   `requirements.txt`, this plan file, or anything at repo root. Copy what you need into
   `experiments/` and iterate on the copy.
3. **Never touch `data/`** beyond reading. Do not write predictions, temp files, or
   anything into `data/audio` or `data/labels`. Do not delete or rename label/audio files.
4. **No version control actions.** Do not run `git add`, `commit`, `push`, `branch`,
   `checkout`, `reset`, `stash`, or `rm` on tracked files. Leave VCS entirely to the human.
5. **Dependency allowlist.** Allowed without asking: anything already in the environment +
   `scikit-learn`, `numpy`, `scipy`, `pandas`, `librosa`, `matplotlib`, `seaborn`, `joblib`.
   `xgboost` / `lightgbm` are allowed **only as CPU wheels** and must be logged as a new dep.
   **Anything requiring a GPU, a large download (>200 MB), or a pretrained model → STOP**,
   log to `experiments/blocked.md`, skip that experiment, continue.
6. **No deletes outside `experiments/`. No network calls** except installing an allowlisted
   pip package. No reaching outside the repo directory.
7. **Compute caps:** hard cap of **12 experiments** total. Per-experiment **timeout 30 min**
   — if exceeded, abort that experiment, log it as `failed: timeout`, continue. If the same
   experiment fails twice, skip it permanently and log why.
8. **Disk hygiene:** save model artifacts only under `experiments/artifacts/`. Keep at most
   the **baseline model + current-best model + most-recent model** as full `.joblib`; for
   others, log metrics only (not the pickled model) to avoid bloat.
9. **Fail loud, never silent.** On any unexpected error, write a full traceback to
   `experiments/errors.log`, mark the experiment failed in the logs, and move on. Do not
   "fix" things by editing files outside `experiments/`.

A human-side safety net is recommended but cannot be enforced from inside: commit or branch
before launching, so the working tree is recoverable.

---

## Definition of "best"
Primary and co-primary, both reported for every experiment:
- **Macro F1** (mean ± std across folds) — overall per-bar quality.
- **Boundary F1** (±1 bar tolerance) and **drop-onset timing error** (median |predicted −
  true| bars for the first `drop` boundary) — these reflect the operational use case
  (timing lights/lasers/pyro), where *when the drop starts* matters more than per-bar labels.

Per-class F1 for **`drop`, `buildUp`, `verse`** is always tracked explicitly (these matter
most and `verse` is the weakest). A config with marginally lower Macro F1 but materially
better drop boundaries may be the recommended pick — call this out in the summary.

Always report trade-offs: training time, inference time, model complexity (low/med/high),
and data sensitivity.

---

## Output & logging structure (create under `experiments/`)
Write/append **immediately after each experiment** (never batch to the end — a crash must
cost at most one experiment).

- `experiments/folds.json` — frozen `track → fold` mapping (seeded). Loaded by every
  experiment so splits never vary.
- `experiments/state.json` — progress + resumption: which experiments are done, current
  cumulative baseline config, current-best config. On startup, **read this first** and skip
  completed experiments.
- `experiments/plan.md` — the ordered experiment queue (write this before running anything
  beyond Step 0; see Process).
- `experiments/results.md` — human-readable, **append-only**, one block per experiment
  (schema below).
- `experiments/log.jsonl` — machine-readable mirror, one JSON object per experiment (full
  params + metrics) for later analysis/plotting.
- `experiments/insights.md` — running notes: surprises, hypotheses, "why did X happen,"
  failure-mode observations, ideas spawned mid-run. **Log anything interesting here**, even
  if it's not a formal result.
- `experiments/artifacts/` — saved models (per disk rule), confusion-matrix images/text,
  per-fold prediction dumps for the baseline and current-best.
- `experiments/blocked.md` — anything skipped due to the safety contract.
- `experiments/errors.log` — tracebacks.
- `experiments/summary.md` — final deliverable.

### Per-experiment log block schema (`results.md` + matching JSON in `log.jsonl`)
```
## EXP-<n>: <short name>
- timestamp, wall-clock duration
- hypothesis: what we expect and why
- variable changed (exactly one — see protocol): <feature added / model / context / smoothing>
- full config: features list, CONTEXT_BARS, n_mels/hop, normalization, model + ALL hyperparams,
  imbalance handling, random seeds
- metrics:
    Macro F1: mean ± std   | per-fold: [...]
    Boundary F1 (±1 bar): mean ± std
    Drop-onset error (bars): median
    Per-class F1: drop / buildUp / verse (+ full table)
    verse support per val fold: [...]   # so low-confidence F1s are visible
- confusion matrix (counts, normalized) — text or path to image
- train time / inference time
- DECISION: keep | revert | neutral — with reason vs significance threshold
- notes / insights
```

---

## Experiment protocol (applies to all)
- **One variable per experiment.** Never combine an untested feature change with an
  untested model/context change. Exception: **per-track normalization** is tested as a
  *modifier* on top of a feature set — log it explicitly as such, not as standalone.
- **Cumulative baseline.** If a change KEEPS (passes the threshold), it becomes the baseline
  for the next experiment. If it REVERTS, restore the prior baseline before continuing.
- **Significance threshold (critical — avoids chasing noise).** With 64 tracks, fold
  variance is real. A change counts as **"keep" only if mean Macro F1 improves by more than
  1 standard deviation across folds**. Otherwise log it as **"neutral"** and revert (prefer
  the simpler config). Apply the same logic to boundary F1 when that's the metric in play.
- **Thin-class watch.** If a change improves overall Macro F1 but *hurts* `verse`/`buildUp`
  specifically, do NOT net it out silently — flag it and let the human judge. Treat `verse`
  results as low-confidence and always log its per-fold support count.
- **Fixed eval scheme** (Step 0) is frozen for the whole run. Same folds, same metric code.
- **Determinism:** fixed seeds for folds and every model. Record them.
- **No leakage:** any global scaler/standardizer and the HMM transition matrix must be fit
  on **training folds only**, per fold. Per-track normalization uses each track's *own*
  stats, so it's leakage-safe across the split — but state this in the log.

---

## Step 0 — Evaluation harness + honest baseline (do this first, then checkpoint)
1. Copy the working pipeline (`scripts/phrase_detection_v1.py` and any needed helpers) into
   `experiments/` and iterate only on the copy.
2. **Detect and log the actual current config verbatim** — especially `CONTEXT_BARS`,
   `N_MELS`, `SR`, hop, and the model + hyperparams — from the copied code. Do **not** assume
   a context width. (History note: this value was 1 in an older draft and later set to 8;
   record whatever the code actually says and treat that as the baseline.)
3. Build the eval harness:
   - **Track-level k-fold CV** (5-fold default). Freeze the fold assignment to `folds.json`.
     Where feasible, arrange folds so each contains at least some `verse`/rare-class tracks;
     if not possible, log which folds lack them.
   - **Macro F1** (mean ± std), full per-class F1, **boundary F1 (±1 bar)** with greedy
     nearest-boundary matching (each true boundary matched to ≤1 predicted), and
     **drop-onset timing error**.
   - **Confusion matrix** logging per experiment.
4. Run the **baseline** (current features + current model) under this scheme and log it as
   `EXP-0`. The old single-split number (Macro F1 0.54) is **not comparable** — do not try to
   reconcile it; this is the new reference point.
5. **Checkpoint:** write `state.json`, print a progress summary, then proceed to write
   `plan.md`.

---

## Candidate features to test (each as its own experiment, vs current cumulative baseline)
Musical reasoning kept from prior analysis:
- **RMS / loudness per bar** — energy is likely more predictive than timbre (drops vs quiet
  are energy-first).
- **Per-track normalization** (modifier) — normalize each bar's features (esp. energy)
  against that track's own median/std, so mastering/loudness differences aren't learned as
  signal.
- **Onset/kick density per bar** — wire in the existing kick detector output. Directly
  separates buildUp/drop/preDropFill.
- **Spectral centroid** (brightness) + **zero-crossing rate** (noisiness) — cheap scalars,
  may help `lead`/`verse`.
- **Position-in-track** (normalized 0–1) — phrases have positional tendencies. **Watch its
  train/val gap** — with one artist this can overfit to that artist's song structure.
- **Delta features** — change vs previous bar; targets `buildUp` (defined by rising
  trajectory, not absolute level).

---

## Suggested ordered queue (write final order to `experiments/plan.md`, justify briefly)
Order chosen by expected impact and cheapness; adjust if Step 0 reveals reasons to.
1. **EXP-0** Baseline (above).
2. **EXP-1** + RMS/energy.
3. **EXP-2** + per-track normalization (modifier on current features).
4. **EXP-3** + onset/kick density.
5. **EXP-4** + spectral centroid + ZCR.
6. **EXP-5** + position-in-track (watch overfitting).
7. **EXP-6** + delta features.
8. **EXP-7** `CONTEXT_BARS=2` as a standalone variable on the current cumulative baseline.
9. **EXP-8** Model swap → **RandomForest** (sklearn, zero new dependency), same feature set,
   same imbalance handling held constant.
10. **EXP-9** (optional, only if RF is promising and budget allows) → XGBoost **or** LightGBM
    (pick one; CPU wheel; log as new dep).
11. **EXP-10** Temporal post-processing: **HMM/Viterbi** decode over predicted class
    probabilities, transition matrix learned per-fold from training label sequences.
12. **EXP-11** (if budget allows) median-smoothing and/or 4-bar boundary snapping; compare
    boundary F1 before/after. Smoothing is expected to move boundary F1 far more than Macro F1.

Out of scope this run (note as future work in summary, **do not implement/benchmark**):
RNN/LSTM/GRU/TCN/transformers and pretrained audio embeddings (e.g. MERT) — data-hungry,
revisit at 100+ labelled tracks.

---

## Process
1. Do Step 0, log `EXP-0`, checkpoint.
2. Write the ordered queue + brief justification to `experiments/plan.md`.
3. Run experiments one at a time. After each: append to `results.md` + `log.jsonl`, update
   `state.json`, save artifacts per disk rule, jot any insight to `insights.md`.
4. Keep/revert per the significance threshold; maintain the cumulative baseline.
5. **Checkpoints:** after Step 0, and after every 2–3 experiments, print a short progress
   summary (done / current best / next up).
6. On reaching the 12-experiment cap or exhausting the queue, **stop** and write
   `experiments/summary.md`:
   - Best configuration found, with full metrics (Macro F1, boundary F1, drop-onset error,
     per-class F1).
   - **Trade-off table:** each major approach × {Macro F1, boundary F1, train time, inference
     time, complexity low/med/high, data sensitivity}.
   - Clear recommendation for what to use going forward, and why — including whether to
     optimize for Macro F1 or for boundary timing given the operational goal.
   - **Caveat to state explicitly:** greedy cumulative feature selection is order-dependent
     and may miss feature pairs that only help together; the result is a strong local
     optimum, not a proven global one.
   - Anything not reached within budget, listed under "not yet tried."
   - Future work (sequence models, pretrained embeddings) once more data exists.

---

## Resumption
On startup, read `state.json`. If a prior run exists, skip completed experiments and resume
from the next pending one using the existing frozen folds. Never re-run completed experiments
(it would waste budget and could change the recorded baseline).
