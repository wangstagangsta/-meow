# train_beat_mvp.py

import json
import math
import os
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------
# CONFIG
# ------------------

TARGET_SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512  # ~11.6 ms at 44.1k
BEAT_TOLERANCE_SEC = 0.03  # +/- 30 ms


# ------------------
# UTILS: LOAD LABEL JSON + GENERATE BEATS
# ------------------

def load_label_json(label_path: str) -> dict:
    with open(label_path, "r") as f:
        return json.load(f)


def generate_beats_from_constant_bpm(
    bpm: float,
    duration: float,
    downbeat_offset_sec: float = 0.0,
) -> List[float]:
    """
    Generate a simple beatgrid for a constant-BPM track.

    bpm: beats per minute
    duration: track duration in seconds
    downbeat_offset_sec: time of the first downbeat in seconds
                         (0 for now, can adjust later)
    """
    if bpm is None or bpm <= 0:
        return []

    period = 60.0 / float(bpm)  # seconds per beat
    beat_times = []

    t = downbeat_offset_sec
    # generate until we go past duration
    while t < duration:
        beat_times.append(t)
        t += period

    return beat_times


# ------------------
# UTILS: AUDIO -> MEL + FRAME TIMES
# ------------------

def load_audio_to_mel(audio_path: str):
    """
    Load mp3/m4a/etc to mono log-mel spectrogram.

    Returns:
        mel_db: np.array, shape (T, N_MELS)  # time-major
        frame_times: np.array, shape (T,)
        duration: float (seconds)
    """
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration = len(y) / sr

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape (N_MELS, T_frames)

    # Transpose to (T, N_MELS)
    mel_db = mel_db.T

    # Frame times (center of each frame)
    frames = np.arange(mel_db.shape[0])
    frame_times = librosa.frames_to_time(
        frames, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    return mel_db, frame_times, duration


# ------------------
# UTILS: BEAT TIMES -> FRAME LABELS
# ------------------

def beat_times_to_frame_labels(
    beat_times: List[float],
    frame_times: np.ndarray,
    tolerance_sec: float = BEAT_TOLERANCE_SEC
) -> np.ndarray:
    """
    Given beat timestamps and frame center times,
    return binary labels per frame (1 if near a beat).
    """
    labels = np.zeros_like(frame_times, dtype=np.float32)

    if len(beat_times) == 0:
        return labels

    beat_idx = 0
    num_beats = len(beat_times)

    for i, ft in enumerate(frame_times):
        # advance beat_idx until closest beat is >= current time
        while beat_idx + 1 < num_beats and beat_times[beat_idx] < ft:
            if abs(beat_times[beat_idx + 1] - ft) < abs(beat_times[beat_idx] - ft):
                beat_idx += 1
            else:
                break
        if abs(beat_times[beat_idx] - ft) <= tolerance_sec:
            labels[i] = 1.0

    return labels


# ------------------
# DATASET
# ------------------

class BeatActivationDataset(Dataset):
    """
    Each item = full track mel + beat labels.
    For MVP: one track per batch.
    """

    def __init__(self, items: List[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_path = item["audio_path"]
        label_path = item["label_path"]

        # 1) Load mel and frame times
        mel_db, frame_times, duration_audio = load_audio_to_mel(audio_path)

        # 2) Load labels from your JSON v2
        label_json = load_label_json(label_path)
        bpm = float(label_json["bpm"])
        duration_label = float(label_json.get("duration", duration_audio))
        downbeat_offset = float(label_json.get("downbeatOffset", 0))

        # Assuming downbeatOffset is in seconds (if it's in ms, divide by 1000)
        # If you later store ms, do:
        # downbeat_offset_sec = downbeat_offset / 1000.0
        downbeat_offset_sec = downbeat_offset

        # 3) Generate beats (constant BPM)
        beat_times = generate_beats_from_constant_bpm(
            bpm=bpm,
            duration=duration_label,
            downbeat_offset_sec=downbeat_offset_sec,
        )

        # 4) Frame labels from beat times
        labels = beat_times_to_frame_labels(beat_times, frame_times)

        # 5) To torch
        mel_tensor = torch.from_numpy(mel_db).float()    # (T, N_MELS)
        labels_tensor = torch.from_numpy(labels).float() # (T,)

        return mel_tensor, labels_tensor


def collate_full_tracks(batch):
    """
    Simple collate: no padding, assumes all tracks similar length.
    For MVP with batch_size=1 this is trivial.
    """
    # batch is list of (mel, labels)
    mels = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    # For now, just stack (works if lengths equal; if not, keep batch_size=1)
    mel_batch = torch.stack(mels, dim=0)      # (B, T, N_MELS)
    label_batch = torch.stack(labels, dim=0)  # (B, T)

    return mel_batch, label_batch


# ------------------
# MODEL: Minimal Beat CRNN
# ------------------

class BeatCRNN(nn.Module):
    """
    Minimal CRNN: (B, T, N_MELS) -> (B, T) beat logits.
    """

    def __init__(self, n_mels=N_MELS, hidden_size=128, num_layers=2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # halve freq

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # halve freq again
        )

        freq_out = n_mels // 4
        cnn_channels = 64
        rnn_input_size = cnn_channels * freq_out

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.output = nn.Linear(hidden_size * 2, 1)  # no sigmoid; we use BCEWithLogits

    def forward(self, x):
        """
        x: (B, T, N_MELS)
        """
        B, T, F = x.shape

        # (B, 1, T, F)
        x = x.unsqueeze(1)

        # CNN: (B, C, T, F')
        x = self.cnn(x)
        B, C, T_new, F_new = x.shape

        # (B, T, C*F')
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T_new, C * F_new)

        # RNN: (B, T, 2H)
        x, _ = self.rnn(x)

        # Output logits: (B, T, 1) -> (B, T)
        logits = self.output(x).squeeze(-1)

        return logits


# ------------------
# TRAINING LOOP
# ------------------

def train_mvp(
    train_items: List[dict],
    val_items: List[dict] = None,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = BeatActivationDataset(train_items)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_full_tracks,
    )

    if val_items is not None and len(val_items) > 0:
        val_ds = BeatActivationDataset(val_items)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_full_tracks,
        )
    else:
        val_loader = None

    model = BeatCRNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for mel_batch, label_batch in train_loader:
            mel_batch = mel_batch.to(device)  # (B, T, F)
            label_batch = label_batch.to(device)  # (B, T)

            optim.zero_grad()
            logits = model(mel_batch)  # (B, T)

            # mask in case shapes mismatch (e.g. T_new < T due to CNN pooling)
            T_pred = logits.shape[1]
            T_true = label_batch.shape[1]
            T_min = min(T_pred, T_true)

            loss = criterion(
                logits[:, :T_min],
                label_batch[:, :T_min],
            )
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch}/{num_epochs} - Train loss: {avg_loss:.4f}")

        # simple val loop
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for mel_batch, label_batch in val_loader:
                    mel_batch = mel_batch.to(device)
                    label_batch = label_batch.to(device)

                    logits = model(mel_batch)
                    T_pred = logits.shape[1]
                    T_true = label_batch.shape[1]
                    T_min = min(T_pred, T_true)

                    loss = criterion(
                        logits[:, :T_min],
                        label_batch[:, :T_min],
                    )
                    val_loss += loss.item()

            avg_val_loss = val_loss / max(1, len(val_loader))
            print(f"           Val loss:   {avg_val_loss:.4f}")

    return model


# ------------------
# MAIN ENTRY (example usage)
# ------------------

if __name__ == "__main__":
    """
    Define your 5 songs here.

    Example structure:
        data/
          audio/
            track1.mp3
            track2.m4a
          labels/
            track1.labels.json
            track2.labels.json
    """
    train_items = [
        {
            "audio_path": "data/audio/track1.mp3",
            "label_path": "data/labels/track1.labels.json",
        },
        {
            "audio_path": "data/audio/track2.m4a",
            "label_path": "data/labels/track2.labels.json",
        },
        # add up to 5
    ]

    # For MVP, you can first:
    #  - train on [train_items[:1]] to overfit one track
    #  - then train on all with maybe one as val

    print("=== Overfitting on 1 track ===")
    model = train_mvp(train_items=train_items[:1], val_items=None, num_epochs=30)

    print("=== Training on all tracks (simple val on last) ===")
    model = train_mvp(
        train_items=train_items[:-1],
        val_items=train_items[-1:],
        num_epochs=30,
    )

    # You can save the model if you want:
    torch.save(model.state_dict(), "beat_crnn_mvp.pth")
