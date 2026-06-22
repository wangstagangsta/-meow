## Goal

  Software that analyses hard dance (Hardstyle) audio and outputs timing data to help
  sync lighting, lasers, visuals, and pyro. Audio is pre-processed offline (not live).

  MVP scope — one specific Hardstyle artist, two features:

  1. **Beatgrid** — BPM + downbeat offset.
  2. **Phrase detector** — given BPM and DOWNBEAT offset, predicts the phrasing of the track.

  Phrase taxonomy (exact label strings used in code/data):
  `quiet`, `verse`, `lead`, `buildUp`, `preDropFill`, `drop`, `bridge`
  (`quiet` covers intro and outro — there is no `intro` label.)


## Current Progress

  - **Beat CRNN** — trained, checkpoints in `models/beat_crnn/` (`beat_crnn_mvp_v1.pth`, `beat_crnn_mvp_v2.pth`). 
  Currently using **v2**. Training: `scripts/train_beat_mvp.py`.
  - **Phrase detector v1** — sklearn MLP over per-bar log-mel
  features (mean+std, 64 mels)
    with ±8 bars of context. Trained in
  `notebooks/phrase_detection_v1.ipynb`
    (`scripts/phrase_detection_v1.py` mirrors it). Saved to
  `models/phrase_detection_v1.joblib`.
    Validation accuracy ~57–76% per track; `verse` is weak (few
  samples).
  - Inference cells in the notebook run the phrase model on a new
  song and export an
    importable label JSON.


## Key conventions & gotchas

  - **`barCount` in label JSON = the LAST bar of that phrase, inclusive** (0-indexed).
    A marker `quiet @ barCount=7` means bars 0–7 (8 bars) are quiet; the next phrase
    starts at bar 8.
  - **Phrase inference depends on accurate BPM + downbeat offset.** Bar boundaries are
    derived from BPM (`60/bpm * 4` per bar); wrong BPM/offset misaligns every feature
    window and wrecks predictions. Model was trained on ~150 BPM Hardstyle, so it
    generalises poorly to very different tempos.
  - Assumes 4/4, 4 beats per bar.

  ## Project structure

  /data
      /audio   - .m4a audio for training; *.infer.json are model predictions (applied in code)
      /labels  - *.labels.json ground-truth labels for training
  /models
      /beat_crnn          - beat CRNN checkpoints (currently using v2)
      phrase_detection_v1.joblib - saved phrase MLP + label encoder
  /notebooks  - exploration + training notebooks
  /scripts    - phrase_detection_v1.py, train_beat_mvp.py
  requirements.txt

  ## Setup

  ```bash
  pip install -r requirements.txt







## blockers/notes

system to organise files
need to add artist names to audio, need to edit all lables? how to sort etc

## next steps

retrain bpm model?
understand phase detection and tune?
more labeling?



