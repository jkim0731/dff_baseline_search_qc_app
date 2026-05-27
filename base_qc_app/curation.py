"""Curation persistence — one row per (session_key, roi_index), overwrites on re-save."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

CURATION_COLUMNS = [
    "user", "session_key", "roi_index", "plane_id", "cell_roi_id",
    "selected", "category", "undecided", "timestamp",
]
DEFAULT_PATH = Path("/root/capsule/scratch/first_try/curation.csv")


def derive_category(selected: list[str], undecided: bool) -> str:
    if undecided:
        return "undecided"
    n = len(selected)
    if n == 0:
        return "none"
    if n == 1:
        return "single"
    return "multiple"


def load_curation(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        for col in CURATION_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[CURATION_COLUMNS]
    return pd.DataFrame(columns=CURATION_COLUMNS)


def save_decision(session_key: str, roi_index: int, plane_id: str, cell_roi_id: int,
                  selected: list[str], undecided: bool, user: str = "",
                  path: Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    df = load_curation(path)
    mask = (df["session_key"] == session_key) & (df["roi_index"] == roi_index)
    df = df.loc[~mask].copy()
    row = {
        "user":        user,
        "session_key": session_key,
        "roi_index":   int(roi_index),
        "plane_id":    plane_id,
        "cell_roi_id": int(cell_roi_id),
        "selected":    ",".join(sorted(selected)),
        "category":    derive_category(selected, undecided),
        "undecided":   bool(undecided),
        "timestamp":   _dt.datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def lookup_decision(df: pd.DataFrame, session_key: str, roi_index: int) -> dict | None:
    rows = df[(df["session_key"] == session_key) & (df["roi_index"] == roi_index)]
    if rows.empty:
        return None
    row = rows.iloc[-1].to_dict()
    sel_str = row.get("selected") or ""
    row["selected_list"] = [s for s in str(sel_str).split(",") if s]
    return row
