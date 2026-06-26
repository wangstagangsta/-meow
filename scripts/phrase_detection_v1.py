"""v1 log-mel phrase pipeline — superseded for modeling, kept for the
``phrase_detection_v1.ipynb`` analysis/training workflow.

Shared primitives (constants, dataclasses, bar/label helpers) now live in
``phrase_core`` and are re-exported here so existing notebook imports keep
working. New code should import them from ``phrase_core`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib.pyplot as plt

from phrase_core import (  # re-exported for backwards compatibility
    PROJECT_ROOT,
    DATA_DIR,
    AUDIO_DIR,
    LABEL_DIR,
    SR,
    N_MELS,
    BEATS_PER_BAR,
    CONTEXT_BARS,
    RANDOM_STATE,
    PhraseMarker,
    TrackLabels,
    BarSegment,
    load_track_labels,
    _bar_duration_seconds,
    _generate_bar_labels,
    build_bar_segments,
    _with_context,
)

plt.style.use("seaborn-v0_8")

np.random.seed(RANDOM_STATE)


def _log_mel_summary(segment: np.ndarray, sr: int, n_mels: int = N_MELS) -> np.ndarray:
    """Return concatenated mean and std of a log-mel spectrogram for a segment."""
    if segment.size == 0 or np.allclose(segment, 0):
        return np.zeros(n_mels * 2, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel + 1e-9)
    return np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)]).astype(np.float32)


def extract_track_features(
    track: TrackLabels,
    audio_dir: Path = AUDIO_DIR,
    sr: int = SR,
    beats_per_bar: int = BEATS_PER_BAR,
) -> Tuple[np.ndarray, List[str], List[BarSegment]]:
    """Compute per-bar features + labels for a single track."""
    audio_path = audio_dir / track.file_name
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file for {track.track_name}: {audio_path}")

    audio, _ = librosa.load(audio_path.as_posix(), sr=sr, mono=True)
    segments = build_bar_segments(track, beats_per_bar)

    features: List[np.ndarray] = []
    labels: List[str] = []
    valid_segments: List[BarSegment] = []
    for seg in segments:
        start_sample = int(seg.start_time * sr)
        end_sample = min(int(seg.end_time * sr), audio.shape[0])
        if start_sample >= audio.shape[0]:
            break
        if end_sample <= start_sample:
            end_sample = min(start_sample + 512, audio.shape[0])
        segment_audio = audio[start_sample:end_sample]
        features.append(_log_mel_summary(segment_audio, sr))
        labels.append(seg.phrase_label)
        valid_segments.append(seg)

    return np.vstack(features), labels, valid_segments


def build_bar_dataset(
    label_dir: Path = LABEL_DIR,
    audio_dir: Path = AUDIO_DIR,
    sr: int = SR,
    beats_per_bar: int = BEATS_PER_BAR,
    context_bars: int = CONTEXT_BARS,
) -> pd.DataFrame:
    """Iterate through labeled tracks and assemble a feature table."""
    records: List[Dict] = []
    label_paths = sorted(label_dir.glob("*.labels.json"))

    for label_path in label_paths:
        track = load_track_labels(label_path)
        track_features, track_labels, segments = extract_track_features(
            track,
            audio_dir=audio_dir,
            sr=sr,
            beats_per_bar=beats_per_bar,
        )
        contextual_features = _with_context(track_features, context=context_bars)

        for feat_vec, label, seg in zip(contextual_features, track_labels, segments):
            records.append(
                {
                    "track": track.track_name,
                    "bar_index": seg.bar_index,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "label": label,
                    "feature": feat_vec,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


def df_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack(df["feature"].to_list())
    y = df["label"].to_numpy()
    return X, y


def train_mlp_classifier(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layers: Tuple[int, ...] = (256, 128),
    random_state: int = RANDOM_STATE,
) -> MLPClassifier:
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        random_state=random_state,
        verbose=False,
    )
    clf.fit(X, y, sample_weight=sample_weights)
    return clf


def evaluate_split(
    model: MLPClassifier,
    X: np.ndarray,
    y_true: np.ndarray,
    split_name: str,
    label_encoder: LabelEncoder,
) -> Dict[str, np.ndarray]:
    y_pred = model.predict(X)
    print(f"\n=== {split_name} ===")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cmap="viridis",
    )
    plt.title(f"{split_name} normalized confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return {"y_pred": y_pred}


def attach_predictions(df: pd.DataFrame, pred_indices: np.ndarray, label_encoder: LabelEncoder) -> pd.DataFrame:
    decoded = label_encoder.inverse_transform(pred_indices)
    enriched = df.copy()
    enriched["pred"] = decoded
    enriched["correct"] = enriched["pred"] == enriched["label"]
    return enriched


def main() -> None:
    bar_df = build_bar_dataset()
    print(f"Loaded {len(bar_df)} bars from {bar_df['track'].nunique()} tracks")
    feature_dim = len(bar_df.iloc[0]["feature"]) if not bar_df.empty else 0
    print(f"Feature dimension (incl. context): {feature_dim}")

    label_counts = bar_df["label"].value_counts().sort_values(ascending=False)
    print("Label counts:")
    print(label_counts)

    tracks = sorted(bar_df["track"].unique())
    train_tracks, val_tracks = train_test_split(
        tracks, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    train_df = bar_df[bar_df["track"].isin(train_tracks)].reset_index(drop=True)
    val_df = bar_df[bar_df["track"].isin(val_tracks)].reset_index(drop=True)

    print(
        f"Train bars: {len(train_df)} across {len(train_tracks)} tracks | "
        f"Val bars: {len(val_df)} across {len(val_tracks)} tracks"
    )

    label_encoder = LabelEncoder().fit(bar_df["label"])
    X_train, y_train = df_to_arrays(train_df)
    X_val, y_val = df_to_arrays(val_df)
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)

    num_classes = len(label_encoder.classes_)
    print(f"Classes ({num_classes}): {label_encoder.classes_}")

    mlp_model = train_mlp_classifier(X_train, y_train_enc)
    train_results = evaluate_split(mlp_model, X_train, y_train_enc, "Train", label_encoder)
    val_results = evaluate_split(mlp_model, X_val, y_val_enc, "Validation", label_encoder)

    val_with_preds = attach_predictions(val_df, val_results["y_pred"], label_encoder)
    val_accuracy_by_phrase = (
        val_with_preds.groupby("label")["correct"].mean().sort_values(ascending=False)
    )
    val_accuracy_by_track = (
        val_with_preds.groupby("track")["correct"].mean().sort_values(ascending=False)
    )

    print("Per-label validation accuracy:")
    print(val_accuracy_by_phrase.to_frame(name="val_accuracy"))

    print("Per-track validation accuracy:")
    print(val_accuracy_by_track.to_frame(name="val_accuracy"))


if __name__ == "__main__":
    main()