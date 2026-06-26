# train_beat_mvp.py

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config, audio->mel preprocessing and the model architecture are the single
# source of truth in beat_core (shared with beat_inference / the backend).
from beat_core import (
    N_MELS,
    BEAT_TOLERANCE_SEC,
    load_audio_to_mel,
    BeatCRNN,
)


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


def discover_label_items(label_dir: str, audio_dir: str) -> List[dict]:
    """
    Pair each label json with its corresponding audio file.
    """
    label_dir_path = Path(label_dir)
    audio_dir_path = Path(audio_dir)

    if not label_dir_path.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir_path}")
    if not audio_dir_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir_path}")

    label_files = sorted(label_dir_path.glob("*.labels.json"))
    if not label_files:
        raise FileNotFoundError(f"No *.labels.json files found in {label_dir_path}")

    items: List[dict] = []
    missing_audio = []
    common_exts = ["", ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aif", ".aiff"]

    for label_path in label_files:
        label_data = load_label_json(str(label_path))
        file_name = label_data.get("fileName")

        candidates = []
        if file_name:
            candidates.append(audio_dir_path / file_name)

        base_name = label_path.stem
        if base_name.endswith(".labels"):
            base_name = base_name[: -len(".labels")]

        for ext in common_exts:
            candidate_path = audio_dir_path / f"{base_name}{ext}"
            if candidate_path not in candidates:
                candidates.append(candidate_path)

        audio_path = next((path for path in candidates if path.exists()), None)

        if audio_path is None:
            missing_audio.append((label_path.name, file_name or f"{base_name}.*"))
            continue

        items.append(
            {
                "audio_path": str(audio_path),
                "label_path": str(label_path),
            }
        )

    if missing_audio:
        print("Warning: Skipping labels with missing audio files:")
        for label_name, expected in missing_audio:
            print(f" - {label_name} (expected audio similar to {expected})")

    if not items:
        raise RuntimeError("No valid label/audio pairs were found.")

    return items


def split_train_val_items(
    items: List[dict], val_count: int
) -> Tuple[List[dict], List[dict]]:
    """
    Split dataset into train/val using the end of the list as validation.
    """
    if val_count <= 0:
        return items, []

    if val_count >= len(items):
        print(
            f"Requested val_count={val_count} but only {len(items)} tracks available. "
            "Reducing validation set so at least one training track remains."
        )
        val_count = max(0, len(items) - 1)

    if val_count == 0:
        return items, []

    train_items = items[:-val_count]
    val_items = items[-val_count:]
    return train_items, val_items


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Beat CRNN MVP model.")
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels",
        help="Directory containing *.labels.json files.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data/audio",
        help="Directory containing audio files referenced by the labels.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=1,
        help="How many tracks to hold out for validation.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional seed to shuffle track order before splitting train/val.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device selection (e.g. cpu, cuda). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="beat_crnn_mvp.pth",
        help="Where to store the trained checkpoint.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the trained checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    items = discover_label_items(args.labels_dir, args.audio_dir)

    if args.shuffle_seed is not None:
        random.seed(args.shuffle_seed)
        random.shuffle(items)

    train_items, val_items = split_train_val_items(items, args.val_count)

    print(f"Found {len(items)} labeled tracks.")
    print(f"Training set size: {len(train_items)}")
    if val_items:
        print(f"Validation set size: {len(val_items)}")
    else:
        print("Validation set size: 0 (validation disabled)")

    model = train_mvp(
        train_items=train_items,
        val_items=val_items,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
    )

    if not args.no_save and args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
