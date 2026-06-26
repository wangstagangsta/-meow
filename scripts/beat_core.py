"""Shared beat-CRNN primitives — single source of truth for the model
architecture and audio->mel preprocessing used by both training and inference.

Kept deliberately small and import-safe (torch + librosa + numpy only, no
Dataset/DataLoader/CLI), so the inference backend imports this + beat_inference
without dragging in the training stack. Both ``train_beat_mvp`` (training) and
``beat_inference`` (serving) import these names from here.
"""

from __future__ import annotations

import warnings

import librosa
import numpy as np
import torch
import torch.nn as nn

# ------------------
# CONFIG  (must match how the checkpoint was trained)
# ------------------

TARGET_SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512  # ~11.6 ms at 44.1k
BEAT_TOLERANCE_SEC = 0.03  # +/- 30 ms

_AUDIO_BACKEND_MESSAGE_SHOWN = False


# ------------------
# AUDIO -> MEL + FRAME TIMES
# ------------------

def load_audio_to_mel(audio_path: str):
    """
    Load mp3/m4a/etc to mono log-mel spectrogram.

    Returns:
        mel_db: np.array, shape (T, N_MELS)  # time-major
        frame_times: np.array, shape (T,)
        duration: float (seconds)
    """
    global _AUDIO_BACKEND_MESSAGE_SHOWN
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    if (
        not _AUDIO_BACKEND_MESSAGE_SHOWN
        and any("PySoundFile failed" in str(w.message) for w in caught)
    ):
        print("Info: soundfile backend unavailable; librosa is using audioread fallback.")
        _AUDIO_BACKEND_MESSAGE_SHOWN = True
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
