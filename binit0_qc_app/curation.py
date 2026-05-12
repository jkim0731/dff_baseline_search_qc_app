"""Curation persistence for the binit0 noise-criterion verification app."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

COLUMNS = [
    "user", "session_key", "roi_index", "plane_id", "cell_roi_id",
    "noise_winner", "visual_best", "verdict", "notes", "timestamp",
]
DEFAULT_PATH = Path("/root/capsule/scratch/binit0_qc_curation.csv")


def load_curation(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save_decision(
    session_key: str, roi_index: int, plane_id: str, cell_roi_id: int,
    noise_winner: str, visual_best: str, verdict: str, notes: str = "",
    user: str = "", path: Path = DEFAULT_PATH,
) -> pd.DataFrame:
    path = Path(path)
    df   = load_curation(path)
    mask = (df["session_key"] == session_key) & (df["roi_index"] == roi_index)
    df   = df.loc[~mask].copy()
    row  = {
        "user":         user,
        "session_key":  session_key,
        "roi_index":    int(roi_index),
        "plane_id":     plane_id,
        "cell_roi_id":  int(cell_roi_id),
        "noise_winner": noise_winner,
        "visual_best":  visual_best,
        "verdict":      verdict,
        "notes":        notes,
        "timestamp":    _dt.datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def lookup_decision(df: pd.DataFrame, session_key: str, roi_index: int) -> dict | None:
    rows = df[(df["session_key"] == session_key) & (df["roi_index"] == roi_index)]
    return rows.iloc[-1].to_dict() if not rows.empty else None
