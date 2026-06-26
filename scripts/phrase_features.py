"""
Canonical phrase-detection feature pipeline (single source of truth for train + inference).

Promoted from the experiments/ research code (EXP-9v winning config). Produces rich per-bar
features and assembles them per a FeatureSpec, identically for training (loop over labelled
tracks) and inference (one track from manual BPM + downbeat offset).

Per-bar raw components:
  logmel_mean (N_MELS), logmel_std (N_MELS)  -- timbre
  rms_mean, rms_std                          -- energy / loudness
  centroid_mean                              -- spectral brightness
  zcr_mean                                   -- noisiness
  onset_density                              -- onsets per second in the bar (kick proxy)

FeatureSpec controls which components + transforms a given model uses. Per-track normalization
is leakage-safe (each track normalized by its own median/std), so it works unchanged at
inference on a single new track.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd

from phrase_core import (
    SR,
    N_MELS,
    BEATS_PER_BAR,
    LABEL_DIR,
    AUDIO_DIR,
    TrackLabels,
    load_track_labels,
    build_bar_segments,
    _with_context,
)

N_FFT = 2048
HOP = 512
SCALAR_FEATURES = ["rms_mean", "rms_std", "centroid_mean", "zcr_mean", "onset_density"]


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    use_logmel: bool = True
    scalar_cols: List[str] = field(default_factory=list)
    per_track_norm: bool = False
    add_delta: bool = False
    add_position: bool = False
    context_bars: int = 8

    def to_dict(self) -> dict:
        return {
            "use_logmel": self.use_logmel,
            "scalar_cols": list(self.scalar_cols),
            "per_track_norm": self.per_track_norm,
            "add_delta": self.add_delta,
            "add_position": self.add_position,
            "context_bars": self.context_bars,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


# The winning configuration (EXP-9v). Default spec for new models.
RECOMMENDED_SPEC = FeatureSpec(
    use_logmel=True,
    scalar_cols=["rms_mean", "rms_std", "onset_density", "centroid_mean", "zcr_mean"],
    per_track_norm=True,
    add_delta=False,
    add_position=False,
    context_bars=8,
)


# ---------------------------------------------------------------------------
# Per-bar raw feature extraction
# ---------------------------------------------------------------------------
def _safe_logmel(seg: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    if seg.size == 0 or np.allclose(seg, 0):
        return np.zeros(N_MELS, np.float32), np.zeros(N_MELS, np.float32)
    n_fft = min(N_FFT, max(256, 2 ** int(np.floor(np.log2(max(seg.size, 1))))))
    mel = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_fft=n_fft, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    logmel = librosa.power_to_db(mel + 1e-9)
    return logmel.mean(axis=1).astype(np.float32), logmel.std(axis=1).astype(np.float32)


def _scalars(seg: np.ndarray, sr: int, n_onsets: int, dur: float) -> dict:
    if seg.size == 0 or np.allclose(seg, 0):
        return {k: 0.0 for k in SCALAR_FEATURES}
    rms = librosa.feature.rms(y=seg, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y=seg, hop_length=HOP)[0]
    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "centroid_mean": float(np.mean(centroid)),
        "zcr_mean": float(np.mean(zcr)),
        "onset_density": float(n_onsets / dur) if dur > 0 else 0.0,
    }


def extract_rich_features(track: TrackLabels, audio_dir: Path = AUDIO_DIR) -> pd.DataFrame:
    """Compute all raw per-bar feature components for a single track (train or inference)."""
    audio_path = audio_dir / track.file_name
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio for {track.track_name}: {audio_path}")

    audio, sr = librosa.load(audio_path.as_posix(), sr=SR, mono=True)
    onset_times = librosa.onset.onset_detect(
        y=audio, sr=sr, hop_length=HOP, units="time", backtrack=False
    )

    records = []
    for seg in build_bar_segments(track):
        s = int(seg.start_time * sr)
        e = min(int(seg.end_time * sr), audio.shape[0])
        if s >= audio.shape[0]:
            break
        if e <= s:
            e = min(s + HOP, audio.shape[0])
        seg_audio = audio[s:e]
        dur = seg.end_time - seg.start_time
        n_onsets = int(np.sum((onset_times >= seg.start_time) & (onset_times < seg.end_time)))
        lm_mean, lm_std = _safe_logmel(seg_audio, sr)
        rec = {
            "track": track.track_name,
            "bar_index": seg.bar_index,
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "label": seg.phrase_label,
            "logmel_mean": lm_mean,
            "logmel_std": lm_std,
        }
        rec.update(_scalars(seg_audio, sr, n_onsets, dur))
        records.append(rec)
    return pd.DataFrame.from_records(records)


def build_rich_dataset(label_dir: Path = LABEL_DIR, audio_dir: Path = AUDIO_DIR,
                       verbose: bool = True) -> pd.DataFrame:
    """Loop over all labelled tracks → combined raw feature table (for training)."""
    frames = []
    label_paths = sorted(label_dir.glob("*.labels.json"))
    for n, lp in enumerate(label_paths, 1):
        track = load_track_labels(lp)
        if not (audio_dir / track.file_name).exists():
            if verbose:
                print(f"  [skip] missing audio: {track.file_name}")
            continue
        frames.append(extract_rich_features(track, audio_dir))
        if verbose:
            print(f"  [{n}/{len(label_paths)}] {track.track_name}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Assembly (raw components -> final feature vectors), per FeatureSpec
# ---------------------------------------------------------------------------
def _base_matrix(tdf: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    mats = []
    if spec.use_logmel:
        mats.append(np.stack(tdf["logmel_mean"].to_list()))
        mats.append(np.stack(tdf["logmel_std"].to_list()))
    for c in spec.scalar_cols:
        mats.append(tdf[c].to_numpy(dtype=np.float32).reshape(-1, 1))
    return np.concatenate(mats, axis=1).astype(np.float32)


def assemble(raw_df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Assemble final per-bar feature vectors. Operates per track (norm/delta/context need it)."""
    out = []
    for track, tdf in raw_df.groupby("track", sort=False):
        tdf = tdf.sort_values("bar_index")
        base = _base_matrix(tdf, spec)

        if spec.per_track_norm:
            med = np.median(base, axis=0)
            std = np.std(base, axis=0) + 1e-8
            base = ((base - med) / std).astype(np.float32)

        feat = base
        if spec.add_delta:
            feat = np.concatenate(
                [feat, np.diff(base, axis=0, prepend=base[:1]).astype(np.float32)], axis=1
            )
        if spec.add_position:
            n = len(tdf)
            pos = (np.arange(n) / max(n - 1, 1)).reshape(-1, 1).astype(np.float32)
            feat = np.concatenate([feat, pos], axis=1)

        feat_ctx = _with_context(feat, context=spec.context_bars)
        for i, (_, row) in enumerate(tdf.iterrows()):
            out.append({
                "track": track,
                "bar_index": int(row["bar_index"]),
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "label": row["label"],
                "feature": feat_ctx[i],
            })
    return pd.DataFrame(out)
