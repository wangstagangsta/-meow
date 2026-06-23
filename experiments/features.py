"""
Rich per-bar feature extraction + caching.

Computes ALL candidate raw features once per bar and caches to disk. Experiments then
select/assemble columns from the cache without re-decoding audio (expensive, 64 .m4a files).

Raw per-bar components stored:
  - logmel_mean (N_MELS), logmel_std (N_MELS)   -- baseline timbre
  - rms_mean, rms_std                            -- energy / loudness
  - centroid_mean                                -- spectral centroid (brightness)
  - zcr_mean                                     -- zero-crossing rate (noisiness)
  - onset_density                                -- onsets per second in the bar (kick proxy)

Position-in-track and delta features are derived at assembly time (they depend on bar
ordering / chosen base columns), see assemble.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phrase_detection_v1 import (  # noqa: E402
    SR,
    N_MELS,
    BEATS_PER_BAR,
    LABEL_DIR,
    AUDIO_DIR,
    load_track_labels,
    build_bar_segments,
)

CACHE_PATH = EXPERIMENTS_DIR / "feature_cache.joblib"
N_FFT = 2048
HOP = 512

# Scalar feature column names (everything except the two log-mel vectors)
SCALAR_FEATURES = ["rms_mean", "rms_std", "centroid_mean", "zcr_mean", "onset_density"]


def _safe_logmel(seg: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
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


def build_cache(force: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force:
        return joblib.load(CACHE_PATH)

    records = []
    label_paths = sorted(LABEL_DIR.glob("*.labels.json"))
    print(f"Building feature cache for {len(label_paths)} tracks...")

    for n, label_path in enumerate(label_paths, 1):
        track = load_track_labels(label_path)
        audio_path = AUDIO_DIR / track.file_name
        if not audio_path.exists():
            print(f"  [skip] missing audio: {audio_path}")
            continue

        audio, sr = librosa.load(audio_path.as_posix(), sr=SR, mono=True)

        # Whole-track onset times (robust) — assign to bars by time window
        onset_times = librosa.onset.onset_detect(
            y=audio, sr=sr, hop_length=HOP, units="time", backtrack=False
        )

        segments = build_bar_segments(track)
        for seg in segments:
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

        print(f"  [{n}/{len(label_paths)}] {track.track_name}: {len(segments)} bars")

    df = pd.DataFrame.from_records(records)
    joblib.dump(df, CACHE_PATH)
    print(f"Cached {len(df)} bars → {CACHE_PATH}")
    return df


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    df = build_cache(force="--force" in sys.argv)
    print(df[["track", "bar_index", "label"] + SCALAR_FEATURES].head(10).to_string())
    print(f"\nTotal: {len(df)} bars, {df['track'].nunique()} tracks")
    print(f"logmel_mean dim: {len(df.iloc[0]['logmel_mean'])}")
