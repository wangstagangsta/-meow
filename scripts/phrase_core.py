"""Shared phrase-pipeline primitives — single source of truth for the data
structures and bar/label helpers used by both training and inference.

Pure-numpy and import-safe: no librosa / sklearn / plotting deps, so the
inference backend can import this without pulling in the analysis stack.
Both ``phrase_detection_v1`` (analysis/notebook) and the serving pipeline
(``phrase_features`` / ``phrase_inference``) import these names from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
LABEL_DIR = DATA_DIR / "labels"

SR = 22050
N_MELS = 64
BEATS_PER_BAR = 4
CONTEXT_BARS = 8
RANDOM_STATE = 42


@dataclass
class PhraseMarker:
    phrase_id: str
    bar_count: int
    time: float


@dataclass
class TrackLabels:
    track_name: str
    file_name: str
    duration: float
    bpm: float
    downbeat_offset: float
    markers: List[PhraseMarker]


@dataclass
class BarSegment:
    track_name: str
    bar_index: int
    start_time: float
    end_time: float
    phrase_label: str


def load_track_labels(label_path: Path) -> TrackLabels:
    """Load a label JSON into a TrackLabels dataclass."""
    with label_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    markers = sorted(
        (
            PhraseMarker(
                phrase_id=marker["phraseId"],
                bar_count=int(marker["barCount"]),
                time=float(marker["time"]),
            )
            for marker in payload.get("phraseMarkers", [])
        ),
        key=lambda m: m.bar_count,
    )

    return TrackLabels(
        track_name=label_path.stem.replace(".labels", ""),
        file_name=payload["fileName"],
        duration=float(payload["duration"]),
        bpm=float(payload["bpm"]),
        downbeat_offset=float(payload["downbeatOffset"]),
        markers=markers,
    )


def _bar_duration_seconds(track: TrackLabels, beats_per_bar: int = BEATS_PER_BAR) -> float:
    beat_duration = 60.0 / track.bpm
    return beat_duration * beats_per_bar


def _generate_bar_labels(track: TrackLabels, beats_per_bar: int = BEATS_PER_BAR) -> List[str]:
    """Return the phrase label for every bar in the track.

    For inference (no markers) every bar is labelled "" — the model fills these in.
    """
    bar_duration = _bar_duration_seconds(track, beats_per_bar)
    total_bars = int(np.ceil(max(track.duration - track.downbeat_offset, 0) / bar_duration))

    labels: List[str] = [""] * total_bars

    for i, marker in enumerate(track.markers):
        start_bar = track.markers[i - 1].bar_count + 1 if i > 0 else 0
        end_bar = marker.bar_count + 1
        for b in range(start_bar, min(end_bar, total_bars)):
            labels[b] = marker.phrase_id

    return labels


def build_bar_segments(track: TrackLabels, beats_per_bar: int = BEATS_PER_BAR) -> List[BarSegment]:
    """Return bar-level segments with phrase labels and timing."""
    bar_labels = _generate_bar_labels(track, beats_per_bar)
    bar_duration = _bar_duration_seconds(track, beats_per_bar)

    segments: List[BarSegment] = []
    for bar_idx, label in enumerate(bar_labels):
        start_time = track.downbeat_offset + bar_idx * bar_duration
        end_time = min(start_time + bar_duration, track.duration)
        segments.append(
            BarSegment(
                track_name=track.track_name,
                bar_index=bar_idx,
                start_time=start_time,
                end_time=end_time,
                phrase_label=label,
            )
        )

    return segments


def _with_context(features: np.ndarray, context: int) -> np.ndarray:
    """Concatenate ±context bars for each bar."""
    if context <= 0:
        return features

    pad = np.zeros((context, features.shape[1]), dtype=features.dtype)
    padded = np.vstack([pad, features, pad])
    window_size = context * 2 + 1
    contextualized = []
    for idx in range(features.shape[0]):
        window = padded[idx : idx + window_size]
        contextualized.append(window.reshape(-1))
    return np.vstack(contextualized)
