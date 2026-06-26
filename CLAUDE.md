## Goal

Software that analyses hard dance (Hardstyle) audio and outputs timing data to help
sync lighting, lasers, visuals, and pyro. Audio is pre-processed offline (not live).

MVP scope — one specific Hardstyle artist, two features:

1. **Beatgrid** — BPM + downbeat offset.
2. **Phrase detector** — given BPM and DOWNBEAT offset, predicts the phrasing of the track.

Phrase taxonomy (exact label strings used in code/data):
`quiet`, `verse`, `lead`, `buildUp`, `preDropFill`, `drop`, `bridge`
(`quiet` covers intro and outro — there is no `intro` label.)

See `ROADMAP.md` for prioritized future plans.


## Workflows — the 3 things this repo is for

The repo has a lot of files/notebooks. Almost everything maps to one of three jobs;
figure out which you're in before touching files.

1. **Testing / experimentation** — try features/models, benchmark, iterate.
   - Phrase: `experiments/` (frozen CV split, harness, tuning, results — READ `experiments/summary.md`
     for what's already been tried). Scratch notebooks: `notebooks/phrase_detection_v1.ipynb`.
   - Beat / kick: `notebooks/train_beat_mvp.ipynb`, `notebooks/kick_detectorv1.ipynb`.

2. **Running models** — inference on a track, interactively.
   - Phrase: `notebooks/phrase_inference.ipynb` (bundle-driven; set knobs, pick `MODEL_PATH`,
     run; exports `*.infer.json`).
   - Beat: `scripts/beat_inference.py` → `load_model()` + `predict_beatgrid()`. BPM + offset come
     from a **comb-filter grid search** over the activation (`estimate_bpm_from_activation`,
     mirroring the `notebooks/train_beat_mvp.ipynb` beatgrid pipeline) — NOT median inter-beat
     interval, which gave noisy non-round BPMs. This file is copied verbatim into the labeler
     backend (`backend/models/beat_inference.py`), so keep the two identical.

3. **Packaging models for the labeler** (the labeling-website backend). Copy *only* these:
   - Phrase: `phrase_core.py`, `phrase_features.py`, `phrase_smoothing.py`, `phrase_inference.py`
     + bundle `models/phrase_detection/phrase_xgb_viterbi.joblib`.
     Deps: numpy, pandas, librosa, joblib, scikit-learn, xgboost.
   - Beat: `beat_core.py`, `beat_inference.py` + checkpoint `models/beat_crnn/beat_crnn_mvp_v2.pth`.
     Deps: torch, librosa, numpy.
   - **Do NOT ship** `phrase_detection_v1.py` or any `train_*.py` — those are training/analysis
     only and pull in seaborn / sklearn / torch Dataset machinery the backend doesn't need.


## Current Progress

- **Beat CRNN** — checkpoints in `models/beat_crnn/` (using **v2**). Train:
  `scripts/train_beat_mvp.py`; run: `scripts/beat_inference.py` (`predict_beatgrid()`).
  Currently predicts *any* beat offset; a future retrain will target the **first downbeat**
  specifically (needed to link with the phrase model) — so the `downbeat_offset` it returns is
  just the first detected beat; verify phase before feeding it into the phrase model.

- **Phrase detector — benchmarked & upgraded (see `experiments/summary.md`).**
  A full 5-fold CV benchmark campaign (64 tracks, 8009 bars) compared features + models.
  - **Best: XGBoost + per-track-normalized log-mel + scalar features (RMS, onset-density,
    spectral-centroid, ZCR) + Viterbi smoothing.** Macro F1 **0.842 ± 0.030**, boundary F1
    0.792, drop-onset error 0 bars. Bundle: `models/phrase_detection/phrase_xgb_viterbi.joblib`.
  - Dependency-free alt: `phrase_rf_viterbi.joblib` (RandomForest, Macro F1 0.82).
  - Old baseline (log-mel MLP, Macro F1 0.69): `phrase_mlp_baseline.joblib`.
  - Per-class: `drop` 0.96 and `quiet` 0.89 are strong; `bridge` 0.67 is weakest (confuses
    with `verse` — partly irreducible label ambiguity). NOTE: `verse` is now *strong* (0.87);
    the old "verse is weak" note was from a tiny 19-track dataset.

- **Inference** — `notebooks/phrase_inference.ipynb` is bundle-driven: set the knobs, pick a
  `MODEL_PATH`, run. Includes a model-comparison cell. Exports importable `*.infer.json`.


## Pipeline architecture (post-benchmark)

Both models follow the same `*_core` pattern: a small, import-safe `_core` module holds the
shared pieces (single source of truth — no train/serve skew), training and inference both
import from it. The `_core` modules deliberately have **no** seaborn/sklearn/Dataset deps so the
labeler backend can import them cleanly (see Workflow 3).

**Phrase pipeline:**
- `scripts/phrase_core.py` — **shared primitives**: constants, `PhraseMarker`/`TrackLabels`/
  `BarSegment` dataclasses, label loading + bar segmentation (`build_bar_segments`, `_with_context`).
- `scripts/phrase_features.py` — rich per-bar feature extraction + assembly + `FeatureSpec`.
- `scripts/phrase_smoothing.py` — Viterbi decoding + transition-matrix learning.
- `scripts/phrase_inference.py` — bundle-driven `predict_track()` + label-JSON export.
- `scripts/train_phrase_model.py` — CLI to train any (model, spec) → standardized bundle.
- `scripts/phrase_detection_v1.py` — older log-mel-only pipeline, **analysis/notebook only**
  (MLP train/eval/plots). It now re-exports the primitives from `phrase_core` for backwards
  compat; superseded for modeling. Do not add it to the backend.

**Beat pipeline:**
- `scripts/beat_core.py` — **shared**: config (`TARGET_SR=44100`, `N_MELS=128`, …),
  `load_audio_to_mel` preprocessing, and the `BeatCRNN` model class.
- `scripts/beat_inference.py` — `load_model()` (rebuilds `BeatCRNN`, loads the state_dict),
  `predict_beatgrid()` (activation → comb-filter grid search over BPM + offset → beat times),
  `to_beatgrid_json()`. Kept identical to the labeler backend's copy.
- `scripts/train_beat_mvp.py` — trainer (Dataset + loop + CLI); imports model + preprocessing
  from `beat_core`. The `.pth` is a **state_dict** (weights only), not a pickled model.

**Model bundle schema** (phrase — what inference + model-swap rely on): a joblib dict with
`model`, `label_encoder`, `spec` (FeatureSpec dict), `classes`, `transition_matrix`,
`apply_viterbi`, `meta`. Because each bundle carries its own spec + transition matrix,
**swapping models = changing the file path**; the right features + smoothing follow.

Train / regenerate / make a variant for comparison:
```bash
python scripts/train_phrase_model.py --model xgb --viterbi --name phrase_xgb_viterbi
python scripts/train_phrase_model.py --model rf  --viterbi --name phrase_rf_viterbi
# flags: --no-scalars --no-norm --position --delta --context N --rebuild
python scripts/train_beat_mvp.py --num-epochs 30 --save-path models/beat_crnn/beat_crnn_mvp.pth
```
(First phrase run extracts audio features and caches to `models/phrase_detection/_feature_cache.joblib`.)


## Key conventions & gotchas

- **`barCount` in label JSON = the LAST bar of that phrase, inclusive** (0-indexed).
  A marker `quiet @ barCount=7` means bars 0–7 (8 bars) are quiet; next phrase starts at bar 8.
- **Phrase inference depends on accurate BPM + downbeat offset.** Bar boundaries derive from
  BPM (`60/bpm * 4` per bar); a *whole-beat* (wrong-phase) downbeat error is catastrophic,
  sub-beat is fine. Trained on ~150 BPM Hardstyle — generalises poorly to very different tempos.
- Assumes 4/4, 4 beats per bar.
- **`onset_density` feature is a librosa-onset proxy**, not the (planned) real kick detector.
- **The two pipelines use different sample rates** — phrase features run at `SR=22050`, the beat
  CRNN at `TARGET_SR=44100`. Each lives in its own `_core.py`, so don't cross-import config.
- Use the project venv: `.venv/bin/python3`.


## Project structure

```
/data
    /audio   - .m4a audio for training; *.infer.json are model predictions (applied in code)
    /labels  - *.labels.json ground-truth labels (cue = first downbeat; being re-labeled)
/models
    /beat_crnn          - beat CRNN checkpoints (using v2)
    /phrase_detection   - phrase model bundles:
        phrase_xgb_viterbi.joblib   (recommended)
        phrase_rf_viterbi.joblib    (dependency-free alt)
        phrase_mlp_baseline.joblib  (old baseline)
        _feature_cache.joblib       (cached audio features)
/experiments  - phrase-detection benchmark campaign (READ as the record of what was tried):
        summary.md / plan.md / results.md / log.jsonl / insights.md
        folds.json (frozen CV split), harness.py, tune_xgb.py, artifacts/
/notebooks    - exploration + training notebooks (phrase_inference.ipynb, phrase_detection_v1.ipynb,
                train_beat_mvp.ipynb, kick_detectorv1.ipynb, ...)
/scripts      - phrase pipeline: phrase_core.py, phrase_features.py, phrase_smoothing.py,
                phrase_inference.py, train_phrase_model.py, phrase_detection_v1.py (analysis-only)
                beat pipeline:   beat_core.py, beat_inference.py, train_beat_mvp.py
                (the *_core.py modules are the import-safe shared SoT for each pipeline)
ROADMAP.md    - prioritized future plans
requirements.txt
```


## Setup

```bash
pip install -r requirements.txt   # plus: xgboost (CPU) for the recommended model
```


## Audio acquisition (yt-dlp)

- Download new tracks WITHOUT a lossy re-encode:
  `yt-dlp --cookies-from-browser chrome -f bestaudio -x --audio-quality 0 -o "..." LINK`
  Avoid `--audio-format aac` — it transcodes YouTube's Opus stream to AAC lossily for no gain.
- **Codec/bitrate is irrelevant to the models:** features run at `SR=22050` (Nyquist 11 kHz)
  and are averaged over ~1.6s bars, so lossy-compression artifacts (which live in discarded
  high frequencies) never reach a feature. Don't optimize audio quality for training data.
- **DO NOT re-download already-labeled tracks.** Re-encodes can change leading padding /
  duration and shift the downbeat offset out of phase, silently breaking label alignment.
- Converting local files to `.wav`/`.flac` speeds loading and avoids the macOS AVFoundation
  permission popups that come from librosa's `.m4a` decode path.


## blockers/notes

system to organise files:
- need to add artist names to audio, need to edit all labels? how to sort etc

## next steps
See `ROADMAP.md`. Near-term: finish relabeling (cue = first downbeat) → re-run phrase
experiments on clean labels → kick-onset model → downbeat detection → link beat + phrase models.
