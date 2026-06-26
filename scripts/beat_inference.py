"""Beat inference for the Beat CRNN.

Load a checkpoint once, run a track through the same preprocessing the model was
trained on to get a per-frame beat activation, then estimate BPM + first-beat
offset with a comb-filter grid search over the activation. This is the beatgrid
pipeline from notebooks/train_beat_mvp.ipynb (robust, gives clean BPMs) — the
earlier median inter-beat-interval approach was noisy and produced non-round BPMs.

Usage:
    from beat_inference import load_model, predict_beatgrid, to_beatgrid_json

    model = load_model("beat_crnn_mvp_v2.pth")          # state_dict checkpoint
    grid  = predict_beatgrid(model, "track.m4a")        # {"bpm", "downbeat_offset", "beat_times", ...}
    out   = to_beatgrid_json("track.m4a", grid)         # JSON-serializable

NOTE: this checkpoint predicts *any* beat, not specifically the first downbeat,
so `downbeat_offset` here is the first detected beat — verify phase before
feeding it into the phrase model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from beat_core import (
    TARGET_SR,
    HOP_LENGTH,
    BeatCRNN,
    load_audio_to_mel,
)

FPS = TARGET_SR / HOP_LENGTH  # activation frames per second (~86.13)

# Defaults tuned for hardstyle (matches the notebook run that was spot-on).
DEFAULT_BPM_MIN = 130.0
DEFAULT_BPM_MAX = 250.0
DEFAULT_BPM_STEP = 0.5


def load_model(path, device: str | None = None) -> BeatCRNN:
    """Instantiate BeatCRNN and load a state_dict checkpoint in eval mode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BeatCRNN()
    state = torch.load(path, map_location=device)
    # tolerate either a raw state_dict or a {"state_dict": ...} wrapper
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_activation(model: BeatCRNN, audio_path: str):
    """Return (activation, frame_times, duration).

    activation: np.array (T,) of per-frame beat probabilities in [0, 1].
    """
    mel_db, frame_times, duration = load_audio_to_mel(str(audio_path))
    device = next(model.parameters()).device
    x = torch.from_numpy(mel_db).float().unsqueeze(0).to(device)  # (1, T, N_MELS)
    logits = model(x).squeeze(0)                                  # (T,)
    activation = torch.sigmoid(logits).cpu().numpy()

    # CNN pools freq only, so time is preserved; guard against any off-by-N anyway.
    n = min(len(activation), len(frame_times))
    return activation[:n], frame_times[:n], duration


def sample_probs_at_times(frame_times, probs, query_times):
    """Return activation probs at the nearest frame for each query time."""
    frame_times = np.asarray(frame_times)
    probs = np.asarray(probs)
    query_times = np.asarray(query_times)

    indices = np.searchsorted(frame_times, query_times, side="left")
    indices = np.clip(indices, 0, len(frame_times) - 1)
    left = np.clip(indices - 1, 0, len(frame_times) - 1)
    use_left = np.abs(frame_times[left] - query_times) < np.abs(
        frame_times[indices] - query_times
    )
    final = np.where(use_left, left, indices)
    return probs[final]


def find_best_offset_from_activation(
    frame_times, probs, bpm, duration, n_offsets: int = 200
) -> tuple[float, float]:
    """Search offsets in [0, period) for the grid that best aligns with activation."""
    period = 60.0 / float(bpm)
    ft = np.asarray(frame_times)
    pb = np.asarray(probs)
    if duration <= 0 or len(ft) == 0:
        return 0.0, -np.inf

    best_offset, best_score = 0.0, -np.inf
    for off in np.linspace(0.0, period, num=n_offsets, endpoint=False):
        grid_times = np.arange(off, duration, period)
        if len(grid_times) == 0:
            continue
        score = sample_probs_at_times(ft, pb, grid_times).mean()
        if score > best_score:
            best_score, best_offset = score, float(off)
    return best_offset, best_score


def estimate_bpm_from_activation(
    frame_times,
    probs,
    duration,
    bpm_min: float = DEFAULT_BPM_MIN,
    bpm_max: float = DEFAULT_BPM_MAX,
    bpm_step: float = DEFAULT_BPM_STEP,
    n_offsets: int = 120,
) -> tuple[float, float]:
    """Grid-search BPM by how well a constant beatgrid aligns with the activation."""
    best_bpm, best_score = None, -np.inf
    for bpm in np.arange(bpm_min, bpm_max + 1e-6, bpm_step):
        if bpm <= 0:
            continue
        _, score = find_best_offset_from_activation(
            frame_times, probs, bpm, duration, n_offsets=n_offsets
        )
        if score > best_score:
            best_score, best_bpm = score, float(bpm)
    return (best_bpm if best_bpm is not None else float(bpm_min)), best_score


def build_beatgrid(offset: float, bpm: float, duration: float) -> np.ndarray:
    """Full list of beat times from the offset, stepping by the beat period."""
    period = 60.0 / float(bpm) if bpm and bpm > 0 else 0.0
    if period <= 0 or duration <= 0:
        return np.array([])
    return np.arange(offset, duration + 1e-6, period)


def predict_beatgrid(
    model: BeatCRNN,
    audio_path: str,
    bpm_min: float = DEFAULT_BPM_MIN,
    bpm_max: float = DEFAULT_BPM_MAX,
    bpm_step: float = DEFAULT_BPM_STEP,
) -> dict:
    """Full beatgrid for one track: bpm + first-beat offset (via grid search) + beat times."""
    activation, frame_times, duration = predict_activation(model, audio_path)

    bpm, bpm_score = estimate_bpm_from_activation(
        frame_times, activation, duration, bpm_min=bpm_min, bpm_max=bpm_max, bpm_step=bpm_step
    )
    offset, _ = find_best_offset_from_activation(frame_times, activation, bpm, duration)
    beat_times = build_beatgrid(offset, bpm, duration)

    return {
        "bpm": round(float(bpm), 3),
        "downbeat_offset": round(float(offset), 4),
        "duration": round(float(duration), 3),
        "bpm_score": round(float(bpm_score), 4),
        "beat_times": [round(float(t), 4) for t in beat_times],
        "activation": activation,   # np.array kept for plotting/debugging
        "frame_times": frame_times,
    }


def to_beatgrid_json(audio_path: str, grid: dict) -> dict:
    """JSON-serializable beatgrid (drops the raw activation arrays)."""
    return {
        "fileName": Path(audio_path).name,
        "bpm": grid["bpm"],
        "downbeatOffset": grid["downbeat_offset"],
        "duration": grid["duration"],
        "beatTimes": grid["beat_times"],
    }
