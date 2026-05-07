"""Compare-mode panel: side-by-side recipe diff + multi-source runs picker.

Layout (top → bottom):

  1. Sources     — list of runs folders; +Add / −Remove / Refresh.
  2. Runs        — checkbox table of every run discovered across the sources.
                   Showing only the high-signal recipe columns.
  3. Differences — auto-computed: one row per recipe parameter that *differs*
                   between the runs you've checked. Cells highlighted where
                   they differ from the first checked run.
  4. Slots       — 4 hard slots (matching keys 1–4 in the main window).
                   "Quick assign" buttons map all checked runs into slots
                   1..N with one click; per-slot kind picker is still there.

``selectionsChanged(dict)`` is emitted whenever a slot mapping changes;
the main window applies the change immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QMessageBox,
    QPushButton, QSplitter, QTableView, QVBoxLayout, QWidget,
)

from .runs import DEFAULT_RUNS_DIR, discover_runs_multi_with_status

SLOT_KEYS    = ("short", "long", "F0trend", "F0")
KIND_OPTIONS = ("F0trend", "F0")

# These are the "signal" columns we always show in the runs list. Anything
# starting with `recipe_` and *not* in here lives in the diff view.
_RUNS_LIST_COLS = (
    "run_id", "slug", "source_dir", "description",
    "recipe_sigma_kind", "recipe_sigma_method",
    "recipe_M_kind", "recipe_M_c_pos", "recipe_M_c_neg",
    "recipe_fluctuations_method", "recipe_fluctuations_mode",
)
# Columns excluded from the diff view (administrative metadata or human-only).
_DIFF_EXCLUDE_COLS = (
    "run_id", "slug", "run_dir", "source_dir", "created_at",
    "n_sessions", "description",
    "recipe_description",       # free-text — always differs by design
    "recipe_schema_version",    # always equal in practice
)

DIFF_HIGHLIGHT = QColor(255, 226, 176)   # warm pale orange


class CompareRunsPanel(QWidget):
    selectionsChanged = pyqtSignal(dict)

    def __init__(self, runs_dirs: list[Path] | Path | None = None,
                 current: Optional[dict] = None, parent=None):
        super().__init__(parent)
        if runs_dirs is None:
            runs_dirs = []
        elif isinstance(runs_dirs, (str, Path)):
            runs_dirs = [Path(runs_dirs)]
        self._sources: list[Path] = [Path(d) for d in runs_dirs]
        self._df: pd.DataFrame = pd.DataFrame()
        self._checked_run_dirs: set[str] = set()
        self._selections: dict[str, tuple[Path, str, str]] = dict(current or {})

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Sections wired into a vertical splitter so the user can resize
        # them based on which view they care about most at the moment.
        self._build_sources_section(root)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        self._build_runs_section(splitter)
        self._build_diff_section(splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

        self._build_slot_section(root)

        # Initial load
        self._refresh_index()
        self._refresh_diff()
        self._refresh_slot_labels()

    # ── Sources section ──────────────────────────────────────────────────────

    def _build_sources_section(self, root: QVBoxLayout) -> None:
        gb = QGroupBox("Sources (runs folders) — pick a folder containing an index.csv "
                       "and NNNN_<slug>/ subfolders, NOT the inputs folder or a single run")
        gl = QHBoxLayout(gb)
        self.sources_list = QListWidget()
        self.sources_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sources_list.setMaximumHeight(60)
        gl.addWidget(self.sources_list, stretch=1)

        col = QVBoxLayout()
        add_btn = QPushButton("＋ Add folder…")
        rm_btn  = QPushButton("− Remove selected")
        rf_btn  = QPushButton("⟳ Refresh")
        for b in (add_btn, rm_btn, rf_btn):
            b.setFixedHeight(22)
            col.addWidget(b)
        col.addStretch()
        gl.addLayout(col)
        root.addWidget(gb)

        add_btn.clicked.connect(self._add_source)
        rm_btn.clicked.connect(self._remove_selected_sources)
        rf_btn.clicked.connect(self._refresh_index)

    def _add_source(self) -> None:
        start = str(DEFAULT_RUNS_DIR) if DEFAULT_RUNS_DIR.exists() else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Add a runs folder", start)
        if not chosen:
            return
        p = Path(chosen)
        if p in self._sources:
            return
        self._sources.append(p)
        self.sources_list.addItem(str(p))
        self._refresh_index()

    def _remove_selected_sources(self) -> None:
        # The list-item text now embeds a status suffix ("<path>   —   <status>")
        # so map back to the index in self._sources by row position.
        rows = sorted({self.sources_list.row(i)
                       for i in self.sources_list.selectedItems()},
                      reverse=True)
        for r in rows:
            if 0 <= r < len(self._sources):
                self._sources.pop(r)
        self._refresh_index()

    # ── Runs list section ────────────────────────────────────────────────────

    def _build_runs_section(self, parent: QWidget) -> None:
        wrap = QWidget(parent)
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("<b>Runs</b> &nbsp; — check the runs you want to compare"))
        hdr.addStretch()
        all_btn = QPushButton("Check all")
        all_btn.setFixedHeight(20)
        none_btn = QPushButton("Uncheck all")
        none_btn.setFixedHeight(20)
        hdr.addWidget(all_btn)
        hdr.addWidget(none_btn)
        layout.addLayout(hdr)

        self.runs_table = QTableView()
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.runs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.runs_table.setSortingEnabled(True)
        self.runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        self.runs_table.setMinimumHeight(120)
        layout.addWidget(self.runs_table, stretch=1)

        all_btn.clicked.connect(lambda: self._set_all_checked(True))
        none_btn.clicked.connect(lambda: self._set_all_checked(False))

    def _populate_runs_table(self) -> None:
        df = self._df
        cols_present = [c for c in _RUNS_LIST_COLS if c in df.columns]
        # Always include a check column at index 0
        model = QStandardItemModel(len(df), 1 + len(cols_present))
        model.setHorizontalHeaderLabels(["✓"] + cols_present)
        for r, (_, row) in enumerate(df.iterrows()):
            chk = QStandardItem()
            chk.setCheckable(True)
            chk.setEditable(False)
            run_dir = str(row.get("run_dir", ""))
            chk.setData(run_dir, Qt.UserRole)
            chk.setCheckState(Qt.Checked if run_dir in self._checked_run_dirs else Qt.Unchecked)
            model.setItem(r, 0, chk)
            for c, col in enumerate(cols_present, start=1):
                v = row.get(col, "")
                txt = "" if pd.isna(v) else str(v)
                # show only the basename for source_dir to keep the column narrow
                if col == "source_dir":
                    txt = Path(txt).name or txt
                item = QStandardItem(txt)
                item.setEditable(False)
                model.setItem(r, c, item)
        # disconnect old listeners (model is replaced; old one is GC'd)
        self.runs_table.setModel(model)
        model.itemChanged.connect(self._on_run_check_toggled)
        # column widths
        self.runs_table.setColumnWidth(0, 28)
        for c, col in enumerate(cols_present, start=1):
            self.runs_table.setColumnWidth(c, 150 if col in ("slug", "description") else 90)

    def _on_run_check_toggled(self, item: QStandardItem) -> None:
        if item.column() != 0:
            return
        run_dir = item.data(Qt.UserRole)
        if not run_dir:
            return
        if item.checkState() == Qt.Checked:
            self._checked_run_dirs.add(run_dir)
        else:
            self._checked_run_dirs.discard(run_dir)
        self._refresh_diff()

    def _set_all_checked(self, on: bool) -> None:
        model = self.runs_table.model()
        if model is None:
            return
        for r in range(model.rowCount()):
            item = model.item(r, 0)
            if item is None:
                continue
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)

    # ── Diff section ─────────────────────────────────────────────────────────

    def _build_diff_section(self, parent: QWidget) -> None:
        wrap = QWidget(parent)
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(
            "<b>Differences</b> &nbsp; — recipe parameters that differ between checked runs"
        ))
        hdr.addStretch()
        self.show_all_chk = QCheckBox("Show all parameters (incl. agreeing)")
        self.show_all_chk.toggled.connect(lambda _: self._refresh_diff())
        hdr.addWidget(self.show_all_chk)
        layout.addLayout(hdr)

        self.diff_table = QTableView()
        self.diff_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.diff_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.diff_table.horizontalHeader().setStretchLastSection(False)
        self.diff_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.diff_table.setMinimumHeight(80)
        layout.addWidget(self.diff_table, stretch=1)

        self.diff_summary = QLabel("")
        self.diff_summary.setStyleSheet("color: #555; font-size: 9pt;")
        layout.addWidget(self.diff_summary)

    def _refresh_diff(self) -> None:
        df = self._df
        if df.empty or not self._checked_run_dirs:
            self.diff_table.setModel(QStandardItemModel(0, 0))
            self.diff_summary.setText(
                "Check at least one run above to see its parameters."
            )
            return
        # Stable order — match the runs-list display order
        checked_idx = [i for i, rd in enumerate(df["run_dir"])
                       if rd in self._checked_run_dirs]
        sub = df.iloc[checked_idx]
        # Parameter rows = recipe_* columns we don't actively exclude
        param_cols = [c for c in df.columns
                      if c.startswith("recipe_") and c not in _DIFF_EXCLUDE_COLS]
        col_labels = [f"{r['run_id']} · {r['slug']}" for _, r in sub.iterrows()]

        rows: list[tuple[str, list[str], bool]] = []
        agree_count = 0
        for p in param_cols:
            vals = [_fmt_val(v) for v in sub[p].tolist()]
            differs = len(set(vals)) > 1
            if differs:
                rows.append((_short_name(p), vals, True))
            else:
                if self.show_all_chk.isChecked():
                    rows.append((_short_name(p), vals, False))
                agree_count += 1

        model = QStandardItemModel(len(rows), 1 + len(col_labels))
        model.setHorizontalHeaderLabels(["parameter"] + col_labels)
        for r, (name, vals, differs) in enumerate(rows):
            name_item = QStandardItem(("★ " if differs else "  ") + name)
            name_item.setEditable(False)
            if differs:
                f = name_item.font(); f.setBold(True); name_item.setFont(f)
            model.setItem(r, 0, name_item)
            base = vals[0]
            for c, v in enumerate(vals, start=1):
                cell = QStandardItem(v)
                cell.setEditable(False)
                if differs and v != base:
                    cell.setBackground(QBrush(DIFF_HIGHLIGHT))
                model.setItem(r, c, cell)
        self.diff_table.setModel(model)
        self.diff_table.setColumnWidth(0, 220)
        for c in range(len(col_labels)):
            self.diff_table.setColumnWidth(1 + c, 140)

        n_diff = sum(1 for *_, d in rows if d)
        self.diff_summary.setText(
            f"{len(self._checked_run_dirs)} run(s) checked · "
            f"{n_diff} differing parameter(s) · "
            f"{agree_count} agreeing parameter(s){'' if self.show_all_chk.isChecked() else ' (hidden)'}"
        )

    # ── Slot section ─────────────────────────────────────────────────────────

    def _build_slot_section(self, root: QVBoxLayout) -> None:
        gb = QGroupBox("Slot assignments — match keys 1–4 in the main window")
        gl = QVBoxLayout(gb)

        # Quick-assign row
        quick = QHBoxLayout()
        quick.addWidget(QLabel("Quick assign checked runs:"))
        self.quick_kind = QComboBox()
        self.quick_kind.addItems(KIND_OPTIONS)
        quick.addWidget(self.quick_kind)
        qa_btn = QPushButton("→ Slots 1..N (same kind)")
        qa_btn.setToolTip(
            "Map the first 4 checked runs to slots 1–4 with the chosen kind."
        )
        qa_pair_btn = QPushButton("→ Slots 1–2 = run·F0trend, 3–4 = run·F0")
        qa_pair_btn.setToolTip(
            "Map checked runs into 4 slots showing F0trend AND F0 of each run."
        )
        reset_btn = QPushButton("Reset all → legacy")
        for b in (qa_btn, qa_pair_btn, reset_btn):
            quick.addWidget(b)
        quick.addStretch()
        gl.addLayout(quick)

        # Per-slot grid (kind picker + label) — same as before, more compact.
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        self._slot_widgets: dict[str, dict] = {}
        for col, key in enumerate(SLOT_KEYS):
            grid.addWidget(QLabel(f"<b>{col + 1}: {key}</b>"), 0, col, Qt.AlignCenter)
            kind_combo = QComboBox()
            kind_combo.addItems(KIND_OPTIONS)
            grid.addWidget(kind_combo, 1, col)
            assign_btn = QPushButton("← Assign first checked")
            grid.addWidget(assign_btn, 2, col)
            clear_btn = QPushButton("Clear (legacy)")
            grid.addWidget(clear_btn, 3, col)
            label_lbl = QLabel("(legacy)")
            label_lbl.setWordWrap(True)
            label_lbl.setStyleSheet("color: #555; font-size: 9pt;")
            grid.addWidget(label_lbl, 4, col, Qt.AlignTop)
            assign_btn.clicked.connect(lambda _, k=key: self._assign_first_checked(k))
            clear_btn.clicked.connect(lambda _, k=key: self._clear_slot(k))
            self._slot_widgets[key] = {"kind_combo": kind_combo, "label_lbl": label_lbl}
        gl.addLayout(grid)
        root.addWidget(gb)

        qa_btn.clicked.connect(self._quick_assign_same_kind)
        qa_pair_btn.clicked.connect(self._quick_assign_pair_kinds)
        reset_btn.clicked.connect(self._reset_all)

    # ── slot mutation helpers ────────────────────────────────────────────────

    def _checked_runs_in_order(self) -> list[pd.Series]:
        if self._df.empty:
            return []
        rows = []
        for _, row in self._df.iterrows():
            if str(row.get("run_dir", "")) in self._checked_run_dirs:
                rows.append(row)
        return rows

    def _label_for(self, row: pd.Series, kind: str) -> str:
        return f"{row.get('run_id', '?')} {row.get('slug', '')} · {kind}"

    def _assign(self, slot_key: str, row: pd.Series, kind: str) -> None:
        run_dir = Path(str(row["run_dir"]))
        self._selections[slot_key] = (run_dir, kind, self._label_for(row, kind))

    def _assign_first_checked(self, slot_key: str) -> None:
        rows = self._checked_runs_in_order()
        if not rows:
            QMessageBox.information(self, "No checked runs",
                                    "Check at least one run in the Runs table first.")
            return
        kind = self._slot_widgets[slot_key]["kind_combo"].currentText()
        self._assign(slot_key, rows[0], kind)
        self._refresh_slot_labels()
        self.selectionsChanged.emit(dict(self._selections))

    def _quick_assign_same_kind(self) -> None:
        rows = self._checked_runs_in_order()
        if not rows:
            QMessageBox.information(self, "No checked runs",
                                    "Check at least one run in the Runs table first.")
            return
        kind = self.quick_kind.currentText()
        for slot_key, row in zip(SLOT_KEYS, rows):
            self._assign(slot_key, row, kind)
        # Clear remaining slots not covered by checked runs
        for slot_key in SLOT_KEYS[len(rows):]:
            self._selections.pop(slot_key, None)
        self._refresh_slot_labels()
        self.selectionsChanged.emit(dict(self._selections))

    def _quick_assign_pair_kinds(self) -> None:
        """Slots 1-2 = first 2 checked × F0trend; slots 3-4 = same × F0.

        If only one run is checked, both halves use that run.
        """
        rows = self._checked_runs_in_order()
        if not rows:
            QMessageBox.information(self, "No checked runs",
                                    "Check at least one run in the Runs table first.")
            return
        # Pad to 2 rows by repeating the first if only one checked
        runs = (rows + [rows[0]])[:2]
        kinds_per_slot = ((SLOT_KEYS[0], runs[0], "F0trend"),
                          (SLOT_KEYS[1], runs[1], "F0trend"),
                          (SLOT_KEYS[2], runs[0], "F0"),
                          (SLOT_KEYS[3], runs[1], "F0"))
        for slot_key, row, kind in kinds_per_slot:
            self._assign(slot_key, row, kind)
        self._refresh_slot_labels()
        self.selectionsChanged.emit(dict(self._selections))

    def _clear_slot(self, slot_key: str) -> None:
        if slot_key not in self._selections:
            return
        self._selections.pop(slot_key, None)
        self._refresh_slot_labels()
        self.selectionsChanged.emit(dict(self._selections))

    def _reset_all(self) -> None:
        if not self._selections:
            return
        self._selections.clear()
        self._refresh_slot_labels()
        self.selectionsChanged.emit(dict(self._selections))

    def _refresh_slot_labels(self) -> None:
        for key, w in self._slot_widgets.items():
            sel = self._selections.get(key)
            if sel is None:
                w["label_lbl"].setText("(legacy)")
                w["label_lbl"].setStyleSheet("color: #555; font-size: 9pt;")
            else:
                _, _, label = sel
                w["label_lbl"].setText(label)
                w["label_lbl"].setStyleSheet(
                    "color: #c5410d; font-size: 9pt; font-weight: bold;"
                )

    # ── full rebuild ─────────────────────────────────────────────────────────

    def _refresh_index(self) -> None:
        self._df, statuses = discover_runs_multi_with_status(self._sources)
        # Re-render the sources_list with per-source diagnostics
        self.sources_list.clear()
        for d in self._sources:
            status = statuses.get(str(d), "?")
            self.sources_list.addItem(f"{d}   —   {status}")
        # drop checks that no longer exist
        valid = set(str(rd) for rd in self._df.get("run_dir", []))
        self._checked_run_dirs &= valid
        self._populate_runs_table()
        self._refresh_diff()

    # ── public API (preserved for app.py) ────────────────────────────────────

    def selections(self) -> dict[str, tuple[Path, str, str]]:
        return dict(self._selections)

    def set_selections(self, sel: dict[str, tuple[Path, str, str]]) -> None:
        self._selections = dict(sel)
        self._refresh_slot_labels()


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_val(v) -> str:
    """Compact, comparable string representation of a recipe leaf value."""
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        # Trim trailing zeros for stable equality comparison
        s = f"{v:g}"
        return s
    return str(v)


def _short_name(col: str) -> str:
    """``recipe_fluctuations_method`` → ``fluctuations.method`` for display."""
    if col.startswith("recipe_"):
        return col[len("recipe_"):].replace("_", ".", 1).replace("_", ".")
    return col
