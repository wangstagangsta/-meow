"""
Bundle-driven phrase inference. Any standardized bundle (see train_phrase_model.py) runs
through the same path — swapping models is just loading a different bundle file.

A bundle carries its own FeatureSpec and (optionally) a Viterbi transition matrix, so the
correct features + post-processing are applied automatically per model.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

from phrase_features import FeatureSpec, extract_rich_features, assemble
from phrase_smoothing import viterbi_decode
from phrase_core import TrackLabels, PhraseMarker, AUDIO_DIR


def load_bundle(path) -> dict:
    return joblib.load(path)


def make_track(audio_path: Path, bpm: float, offset: float) -> TrackLabels:
    """Build a TrackLabels from manual BPM + downbeat offset for an unlabelled song."""
    dur = librosa.get_duration(path=str(audio_path))
    return TrackLabels(
        track_name=audio_path.stem,
        file_name=audio_path.name,
        duration=dur,
        bpm=bpm,
        downbeat_offset=offset,
        markers=[PhraseMarker(phrase_id="unknown", bar_count=0, time=offset)],
    )


def _aligned_proba(model, X: np.ndarray, classes: list) -> np.ndarray:
    """Map model.predict_proba columns onto the full ordered class list."""
    raw = model.predict_proba(X)
    proba = np.zeros((raw.shape[0], len(classes)), dtype=np.float64)
    for col, cls_enc in enumerate(model.classes_):
        proba[:, int(cls_enc)] = raw[:, col]
    return proba


def predict_track(bundle: dict, track: TrackLabels, audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """Run a bundle on one track. Returns per-bar DataFrame with predictions + probabilities."""
    spec = FeatureSpec.from_dict(bundle["spec"])
    classes = bundle["classes"]
    le = bundle["label_encoder"]

    raw = extract_rich_features(track, audio_dir)
    bar_df = assemble(raw, spec).sort_values("bar_index").reset_index(drop=True)
    X = np.stack(bar_df["feature"].to_list())

    proba = _aligned_proba(bundle["model"], X, classes)

    if bundle.get("apply_viterbi") and bundle.get("transition_matrix") is not None:
        pred_idx = viterbi_decode(proba, np.asarray(bundle["transition_matrix"]))
    else:
        pred_idx = np.argmax(proba, axis=1)

    preds = le.inverse_transform(pred_idx)
    return pd.DataFrame({
        "bar": bar_df["bar_index"].to_numpy(),
        "start": bar_df["start_time"].to_numpy(),
        "end": bar_df["end_time"].to_numpy(),
        "prediction": preds,
        "confidence": proba[np.arange(len(proba)), pred_idx],
    })


def markers_from_predictions(pred_df: pd.DataFrame) -> list:
    """Run-length encode bar predictions into phrase markers (marker at last bar of each run)."""
    preds = pred_df["prediction"].to_list()
    starts = pred_df["start"].to_list()
    bars = pred_df["bar"].to_list()
    markers = []
    for i, p in enumerate(preds):
        if i == len(preds) - 1 or preds[i + 1] != p:
            markers.append({
                "phraseId": p,
                "time": float(starts[i]),
                "barCount": int(bars[i]),
                "msSinceStart": round(float(starts[i]) * 1000),
            })
    return markers


def to_label_json(pred_df: pd.DataFrame, track: TrackLabels) -> dict:
    return {
        "version": 3,
        "fileName": track.file_name,
        "duration": track.duration,
        "bpm": track.bpm,
        "downbeatOffset": track.downbeat_offset,
        "cueTime": 0,
        "quantizeEnabled": True,
        "phraseMarkers": markers_from_predictions(pred_df),
    }
