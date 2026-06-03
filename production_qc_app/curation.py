"""QC curation persistence for production pipeline output."""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("/scratch/production_qc.csv")

DFF_QUALITY_OPTIONS = ["good", "initial bleaching issue", "OK", "bad"]
QC_LABEL_OPTIONS    = ["good", "OK", "ambiguous", "bad"]

COLUMNS = [
    "session", "plane_id", "roi_index",
    "dff_quality",    # one of DFF_QUALITY_OPTIONS or ""
    "qc_label",       # one of QC_LABEL_OPTIONS or ""
    "timestamp",
]


def load_curation(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path, dtype=str)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save_decision(
    session: str,
    plane_id: str,
    roi_index: int,
    dff_quality: str,
    qc_label: str,
    path: Path = DEFAULT_PATH,
) -> pd.DataFrame:
    path = Path(path)
    df = load_curation(path)
    mask = (
        (df["session"] == session)
        & (df["plane_id"] == plane_id)
        & (df["roi_index"] == str(roi_index))
    )
    df = df.loc[~mask].copy()
    row = {
        "session":     session,
        "plane_id":    plane_id,
        "roi_index":   str(roi_index),
        "dff_quality": dff_quality,
        "qc_label":    qc_label,
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def lookup(
    df: pd.DataFrame, session: str, plane_id: str, roi_index: int
) -> dict | None:
    rows = df[
        (df["session"] == session)
        & (df["plane_id"] == plane_id)
        & (df["roi_index"] == str(roi_index))
    ]
    return rows.iloc[-1].to_dict() if not rows.empty else None
