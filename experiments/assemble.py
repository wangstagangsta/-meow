"""
Assemble final per-bar feature vectors from the cached raw features, per a feature spec.

A spec controls one experiment's feature set:
  use_logmel:     include the 128-dim log-mel mean+std
  scalar_cols:    list of scalar feature names to include (from features.SCALAR_FEATURES)
  per_track_norm: z-normalize all base columns using each track's own median/std (leakage-safe)
  add_delta:      append Δ(base vs previous bar)
  add_position:   append normalized bar position in track [0,1]
  context_bars:   ± bars of context concatenation (the existing _with_context)

Operations are applied per track, in order: base -> norm -> delta/position -> context.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR.parent / "scripts"))

from phrase_detection_v1 import _with_context  # noqa: E402


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


def _base_matrix(tdf: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    mats = []
    if spec.use_logmel:
        mats.append(np.stack(tdf["logmel_mean"].to_list()))
        mats.append(np.stack(tdf["logmel_std"].to_list()))
    for c in spec.scalar_cols:
        mats.append(tdf[c].to_numpy(dtype=np.float32).reshape(-1, 1))
    return np.concatenate(mats, axis=1).astype(np.float32)


def assemble(cache_df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = []
    for track, tdf in cache_df.groupby("track", sort=False):
        tdf = tdf.sort_values("bar_index")
        base = _base_matrix(tdf, spec)

        if spec.per_track_norm:
            med = np.median(base, axis=0)
            std = np.std(base, axis=0) + 1e-8
            base = ((base - med) / std).astype(np.float32)

        feat = base
        if spec.add_delta:
            delta = np.diff(base, axis=0, prepend=base[:1]).astype(np.float32)
            feat = np.concatenate([feat, delta], axis=1)
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


def feature_dim(cache_df: pd.DataFrame, spec: FeatureSpec) -> int:
    sample = cache_df[cache_df["track"] == cache_df.iloc[0]["track"]]
    return len(assemble(sample, spec).iloc[0]["feature"])
