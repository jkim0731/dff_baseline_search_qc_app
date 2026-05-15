"""Interactive ROI selection panel — pop-up for the binit0 QC app.

Shows four views (2 scatter, 2 histogram) coloured by winner combo.
Lasso or blob selection writes a CSV to /root/capsule/scratch/example_csv/.
"""
from __future__ import annotations

import datetime as _dt
import re as _re
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QButtonGroup, QComboBox, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
)

from .data import COMBO_KEY, COMBO_KEY_LIST, COMBOS, TARGET_COEF

# ── palette ───────────────────────────────────────────────────────────────────

_COLORS = {
    "c23": "#ff7f0e", "c24": "#d62728", "c25": "#9467bd",
    "c33": "#17becf", "c34": "#bcbd22", "c35": "#e377c2",
    "c44": "#7f7f7f", "c45": "#8c564b",
}
_SHORTCUT = {
    "c23": "3", "c24": "4", "c25": "5",
    "c33": "6", "c34": "7", "c35": "8",
    "c44": "9", "c45": "0",
}
_KEY_TO_COMBO = {
    Qt.Key_3: "c23", Qt.Key_4: "c24", Qt.Key_5: "c25",
    Qt.Key_6: "c33", Qt.Key_7: "c34", Qt.Key_8: "c35",
    Qt.Key_9: "c44", Qt.Key_0: "c45",
}

# ── plot definitions ──────────────────────────────────────────────────────────
# (display label, "scatter"|"hist", x-column, y-column or None)

_PLOT_DEFS = [
    ("SNR vs Skewness",             "scatter", "skewness",   "snr"),
    ("Bleaching vs Sustained",      "scatter", "bleaching",  "sustained"),
    ("Histogram: log₁₀(med-neg / 0.674σ)", "hist", "log_ratio",      None),
    ("Histogram: log₁₀(‖b‖ / F-mean)",    "hist", "log_param_norm", None),
]

_SAVE_DIR    = Path("/scratch/example_csv")
_N_BINS      = 100
_SUBJECT_RE  = _re.compile(r"^(\d+)_")

def _extract_subject(session_key: str) -> str:
    m = _SUBJECT_RE.match(session_key)
    return m.group(1) if m else session_key


# ── data loading ──────────────────────────────────────────────────────────────

def build_select_df(
    sessions: list[tuple[str, Path]],
    combo_runs: dict,
) -> pd.DataFrame:
    """Per-ROI DataFrame: session_key, roi_index, winner_key, metrics, ratio, param_norm."""
    rows: list[dict] = []

    for sess_key, inp_dir in sessions:
        inp = Path(inp_dir) / sess_key
        noise_path = inp / "F_noise.npy"
        if not noise_path.exists():
            continue
        noise  = np.load(noise_path)
        n_rois = len(noise)

        def _arr(name: str) -> np.ndarray:
            p = inp / name
            return np.load(p) if p.exists() else np.full(n_rois, np.nan)

        snr       = _arr("F_snr.npy")
        skewness  = _arr("F_skewness.npy")
        bleaching = _arr("bleaching_metric.npy")
        sustained = _arr("sustained_metric.npy")

        f_path = inp / "F_all_array.npy"
        F_mean = (np.nanmean(np.load(f_path, mmap_mode="r"), axis=1)
                  if f_path.exists() else np.ones(n_rois))

        med_neg: dict[str, np.ndarray] = {}
        res_all: dict[str, np.ndarray] = {}
        for combo in COMBOS:
            ck  = COMBO_KEY[combo]
            run = Path(combo_runs[combo]) / sess_key
            mnp = run / "med_neg_residuals_F0trend.npy"
            rap = run / "res_all.npy"
            med_neg[ck] = np.load(mnp) if mnp.exists() else np.full(n_rois, np.nan)
            res_all[ck] = (np.load(rap) if rap.exists()
                           else np.full((n_rois, 7), np.nan))

        target = TARGET_COEF * noise

        for i in range(n_rois):
            # winner = combo whose med_neg is closest to target
            best_dist, winner = float("inf"), None
            for ck in COMBO_KEY_LIST:
                val = float(med_neg[ck][i])
                if np.isfinite(val) and abs(val - target[i]) < best_dist:
                    best_dist = abs(val - target[i])
                    winner = ck
            if winner is None:
                continue

            mn    = float(med_neg[winner][i])
            ratio = mn / target[i] if target[i] > 1e-9 else np.nan

            r   = res_all[winner][i]        # [b_inf, b_slow, b_fast, b_bright, ...]
            b_n = float(np.sqrt(r[0]**2 + r[1]**2 + r[2]**2 + r[3]**2))
            fm  = float(F_mean[i])
            param_norm = b_n / fm if (np.isfinite(fm) and abs(fm) > 1e-6) else np.nan

            rows.append({
                "session_key": sess_key,
                "roi_index":   i,
                "winner_key":  winner,
                "subject_id":  _extract_subject(sess_key),
                "snr":         float(snr[i]),
                "skewness":    float(skewness[i]),
                "bleaching":   float(bleaching[i]),
                "sustained":   float(sustained[i]),
                "ratio":           ratio,
                "param_norm":      param_norm,
                "log_ratio":       np.log10(ratio)      if (ratio      is not None and np.isfinite(ratio)      and ratio      > 0) else np.nan,
                "log_param_norm":  np.log10(param_norm) if (param_norm is not None and np.isfinite(param_norm) and param_norm > 0) else np.nan,
            })

    cols = ["session_key", "roi_index", "winner_key", "subject_id",
            "snr", "skewness", "bleaching", "sustained",
            "ratio", "param_norm", "log_ratio", "log_param_norm"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ── geometry helpers ──────────────────────────────────────────────────────────

def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: list) -> np.ndarray:
    """Vectorised even-odd ray test.  poly = [(x0,y0), (x1,y1), ...]"""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    inside = np.zeros(len(xs), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        crosses = (yi > ys) != (yj > ys)
        denom   = yj - yi
        x_int   = (xi + (xj - xi) * (ys - yi) / denom) if denom != 0.0 \
                  else np.full_like(ys, np.inf)
        inside ^= crosses & (xs < x_int)
        j = i
    return inside


def _hex_rgb(hex_c: str) -> tuple[int, int, int]:
    h = hex_c.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ── custom ViewBox (lasso + blob) ─────────────────────────────────────────────

class _SelectViewBox(pg.ViewBox):
    lasso_done = pyqtSignal(object)         # list of (x, y)
    blob_click = pyqtSignal(float, float)   # data-coords (cx, cy)
    blob_hover = pyqtSignal(float, float)   # data-coords (cx, cy)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sel_mode = "lasso"
        self._lasso_pts: list[tuple[float, float]] = []

        self._lasso_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(255, 136, 0, 200), width=1, style=Qt.DashLine))
        self.addItem(self._lasso_curve)

        self._blob_circle = pg.PlotCurveItem(
            pen=pg.mkPen(color=(60, 60, 200, 180), width=1, style=Qt.DashLine))
        self._blob_circle.setVisible(False)
        self._blob_circle.setZValue(30)   # above spots (10) and sel highlight (20)
        self.addItem(self._blob_circle)

        self.setMouseMode(pg.ViewBox.PanMode)
        self.setAcceptHoverEvents(True)

    def set_sel_mode(self, mode: str) -> None:
        self._sel_mode = mode
        self._lasso_pts.clear()
        self._lasso_curve.setData([], [])
        self._blob_circle.setData([], [])
        self._blob_circle.setVisible(mode == "blob")

    def hoverEvent(self, ev):
        if self._sel_mode == "blob" and not ev.isExit():
            pt = self.mapSceneToView(ev.scenePos())
            self.blob_hover.emit(float(pt.x()), float(pt.y()))
        else:
            self._blob_circle.setData([], [])

    def mouseDragEvent(self, ev, axis=None):
        if self._sel_mode == "lasso" and ev.button() == Qt.LeftButton:
            ev.accept()
            pt = self.mapSceneToView(ev.scenePos())
            if ev.isStart():
                self._lasso_pts = [(pt.x(), pt.y())]
            else:
                self._lasso_pts.append((pt.x(), pt.y()))
            if len(self._lasso_pts) > 1:
                xs = [p[0] for p in self._lasso_pts] + [self._lasso_pts[0][0]]
                ys = [p[1] for p in self._lasso_pts] + [self._lasso_pts[0][1]]
                self._lasso_curve.setData(xs, ys)
            if ev.isFinish():
                pts = list(self._lasso_pts)
                self._lasso_pts.clear()
                self._lasso_curve.setData([], [])
                if len(pts) > 2:
                    self.lasso_done.emit(pts)
        else:
            super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        if self._sel_mode == "blob" and ev.button() == Qt.LeftButton:
            ev.accept()
            pt = self.mapSceneToView(ev.scenePos())
            self.blob_click.emit(float(pt.x()), float(pt.y()))
        else:
            super().mouseClickEvent(ev)


# ── viewport-level blob click filter ─────────────────────────────────────────

class _ViewportBlobFilter(QObject):
    """Intercepts left-clicks at the QGraphicsView viewport level so they always
    reach blob-selection logic, bypassing pyqtgraph's item event routing which
    may not propagate clicks through ScatterPlotItems to the ViewBox."""
    clicked = pyqtSignal(float, float)   # emits data-space (x, y)

    def __init__(self, view: pg.PlotWidget, vb: pg.ViewBox, parent=None):
        super().__init__(parent)
        self._view   = view
        self._vb     = vb
        self._active = False

    def setActive(self, active: bool):
        self._active = active

    def eventFilter(self, obj, ev):
        if (self._active
                and ev.type() == QEvent.MouseButtonPress
                and ev.button() == Qt.LeftButton):
            scene_pos = self._view.mapToScene(ev.pos())
            data_pos  = self._vb.mapSceneToView(scene_pos)
            self.clicked.emit(float(data_pos.x()), float(data_pos.y()))
            return True     # consume — prevent pyqtgraph from starting a pan
        return False


# ── main window ───────────────────────────────────────────────────────────────

class SelectExamplesWindow(QMainWindow):
    def __init__(self, sessions, combo_runs, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Select Examples")
        self.resize(960, 680)

        self._sessions   = sessions
        self._combo_runs = combo_runs
        self._df: pd.DataFrame | None = None
        self._selected: set[int] = set()           # df row indices
        self._visible: set[str]  = set(COMBO_KEY_LIST)

        # unique subject IDs sorted numerically
        _seen = dict.fromkeys(_extract_subject(sk) for sk, _ in sessions)
        self._subject_ids: list[str] = sorted(
            _seen, key=lambda s: int(s) if s.isdigit() else s)
        self._visible_subjects: set[str] = set(self._subject_ids)

        self._build_ui()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._load_data)
        QTimer.singleShot(0, self._reposition_subject_overlay)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # row 1: combo toggle buttons + clear/all
        row1 = QHBoxLayout()
        self._combo_btns: dict[str, QPushButton] = {}
        for ck in COMBO_KEY_LIST:
            r, g, b = _hex_rgb(_COLORS[ck])
            btn = QPushButton(f"{_SHORTCUT[ck]}: {ck}")
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedHeight(22)
            btn.setStyleSheet(
                f"QPushButton:checked  {{ background: rgb({r},{g},{b}); color: white; }}"
                f"QPushButton:!checked {{ background: #444; color: #888; }}"
            )
            btn.toggled.connect(lambda chk, k=ck: self._on_combo_toggle(k, chk))
            self._combo_btns[ck] = btn
            row1.addWidget(btn)
        row1.addSpacing(8)
        _clear_all_btn = QPushButton("Clear (Z)")
        _clear_all_btn.setFixedHeight(22)
        _clear_all_btn.clicked.connect(self._clear_combos)
        row1.addWidget(_clear_all_btn)
        _all_btn = QPushButton("All (A)")
        _all_btn.setFixedHeight(22)
        _all_btn.clicked.connect(self._select_all_combos)
        row1.addWidget(_all_btn)
        row1.addStretch()
        root.addLayout(row1)

        # row 2: plot selector | Lasso/Blob | size spinbox | status
        row2 = QHBoxLayout()
        self._plot_combo = QComboBox()
        for label, *_ in _PLOT_DEFS:
            self._plot_combo.addItem(label)
        self._plot_combo.currentIndexChanged.connect(self._on_plot_changed)
        row2.addWidget(self._plot_combo)

        row2.addSpacing(12)
        self._lasso_radio = QRadioButton("Lasso")
        self._blob_radio  = QRadioButton("Blob")
        self._lasso_radio.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._lasso_radio)
        bg.addButton(self._blob_radio)
        self._lasso_radio.toggled.connect(self._on_mode_changed)
        self._blob_radio.toggled.connect(self._on_mode_changed)
        row2.addWidget(self._lasso_radio)
        row2.addWidget(self._blob_radio)

        row2.addSpacing(8)
        row2.addWidget(QLabel("Size:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 100)
        self._size_spin.setValue(1)
        self._size_spin.setSingleStep(1)
        self._size_spin.setFixedWidth(55)
        self._size_spin.valueChanged.connect(self._on_size_changed)
        row2.addWidget(self._size_spin)

        row2.addSpacing(8)
        self._home_btn = QPushButton("Home (H)")
        self._home_btn.setFixedHeight(22)
        self._home_btn.clicked.connect(self._home)
        row2.addWidget(self._home_btn)

        row2.addStretch()
        self._status_lbl = QLabel("Loading…")
        row2.addWidget(self._status_lbl)
        root.addLayout(row2)

        # plot widget
        self._vb = _SelectViewBox()
        self._vb.lasso_done.connect(self._on_lasso)
        self._vb.blob_click.connect(self._on_blob)
        self._vb.blob_hover.connect(self._on_blob_hover)
        pw = pg.PlotWidget(viewBox=self._vb)
        pw.setBackground("w")
        self._pw = pw
        self._pi = pw.getPlotItem()
        root.addWidget(pw, stretch=1)

        # one ScatterPlotItem per combo (for scatter plots)
        self._scat: dict[str, pg.ScatterPlotItem] = {}
        for ck in COMBO_KEY_LIST:
            r, g, b = _hex_rgb(_COLORS[ck])
            sp = pg.ScatterPlotItem(
                size=3, pen=pg.mkPen(None), brush=pg.mkBrush(r, g, b, 170))
            sp.setZValue(10)
            sp.setAcceptedMouseButtons(Qt.NoButton)
            self._pi.addItem(sp)
            self._scat[ck] = sp

        # selection highlight
        self._sel_sp = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen("k", width=2), brush=pg.mkBrush(None))
        self._sel_sp.setZValue(20)
        self._sel_sp.setAcceptedMouseButtons(Qt.NoButton)
        self._pi.addItem(self._sel_sp)

        # subject ID overlay (inside the plot, top-right corner)
        self._subject_btns: dict[str, QPushButton] = {}
        overlay = QWidget(self._pw)
        overlay.setObjectName("subjectOverlay")
        overlay.setStyleSheet(
            "QWidget#subjectOverlay { background: rgba(245,245,245,210);"
            " border: 1px solid #bbb; border-radius: 4px; }"
            "QPushButton { font-size: 9pt; padding: 1px 6px; }"
            "QPushButton:checked  { background: #444; color: white; }"
            "QPushButton:!checked { background: #ddd; color: #999; }"
        )
        ov_lay = QVBoxLayout(overlay)
        ov_lay.setContentsMargins(4, 4, 4, 4)
        ov_lay.setSpacing(2)
        for sid in self._subject_ids:
            btn = QPushButton(sid)
            btn.setCheckable(True); btn.setChecked(True)
            btn.setFixedHeight(20)
            btn.toggled.connect(lambda chk, s=sid: self._on_subject_toggle(s, chk))
            self._subject_btns[sid] = btn
            ov_lay.addWidget(btn)
        overlay.adjustSize()
        self._subject_overlay = overlay
        self._pw.installEventFilter(self)

        # viewport-level blob-click filter (reliable click interception)
        self._blob_filter = _ViewportBlobFilter(self._pw, self._vb, parent=self)
        self._pw.viewport().installEventFilter(self._blob_filter)
        self._blob_filter.clicked.connect(self._on_blob)

        # bar items (recreated on each histogram switch)
        self._bars: dict[str, pg.BarGraphItem | None] = {ck: None for ck in COMBO_KEY_LIST}
        self._sel_bars: list[pg.BarGraphItem] = []
        self._hist_bins: np.ndarray | None = None  # shared bin edges for current hist

        # bottom buttons
        row3 = QHBoxLayout()
        self._clear_btn = QPushButton("Clear Selection (C)")
        self._save_btn  = QPushButton("Save Selection… (S)")
        self._save_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._clear_selection)
        self._save_btn.clicked.connect(self._save)
        row3.addStretch()
        row3.addWidget(self._clear_btn)
        row3.addWidget(self._save_btn)
        root.addLayout(row3)

    # ── data ─────────────────────────────────────────────────────────────────

    def _load_data(self):
        self._status_lbl.setText("Building dataset…")
        QWidget.update(self)         # nudge repaint before blocking call
        try:
            self._df = build_select_df(self._sessions, self._combo_runs)
            self._status_lbl.setText(f"{len(self._df)} ROIs loaded")
        except Exception as exc:
            self._status_lbl.setText(f"Error: {exc}")
            return
        self._on_plot_changed()

    # ── event handlers ────────────────────────────────────────────────────────

    def _vis_df(self) -> pd.DataFrame:
        """DataFrame rows that pass both combo and subject filters."""
        df = self._df
        mask = df["winner_key"].isin(self._visible)
        if "subject_id" in df.columns:
            mask &= df["subject_id"].isin(self._visible_subjects)
        return df[mask]

    def _on_combo_toggle(self, key: str, checked: bool):
        if checked:
            self._visible.add(key)
        else:
            self._visible.discard(key)
        self._refresh_display()

    def _on_subject_toggle(self, sid: str, checked: bool):
        if checked:
            self._visible_subjects.add(sid)
        else:
            self._visible_subjects.discard(sid)
        self._refresh_display()

    def _clear_combos(self):
        for btn in self._combo_btns.values():
            btn.setChecked(False)

    def _select_all_combos(self):
        for btn in self._combo_btns.values():
            btn.setChecked(True)

    def _on_mode_changed(self):
        blob = self._blob_radio.isChecked()
        self._vb.set_sel_mode("blob" if blob else "lasso")
        self._blob_filter.setActive(blob)

    def _on_size_changed(self, _value: int):
        self._vb._blob_circle.setData([], [])   # stale; redrawn on next hover

    def _blob_radii(self, kind: str) -> tuple[float, float]:
        """Blob half-extents in data coords, scaled to current view range."""
        size = self._size_spin.value()
        vr   = self._vb.viewRange()             # [[xmin,xmax],[ymin,ymax]]
        x_range = abs(vr[0][1] - vr[0][0]) or 1.0
        y_range = abs(vr[1][1] - vr[1][0]) or 1.0
        if kind == "scatter":
            return size * x_range / 100.0, size * y_range / 100.0
        else:
            # histogram: size=1 → half-extent = bw/2 so [cx-bw/2, cx+bw/2] = one column
            bw = (self._hist_bins[1] - self._hist_bins[0]) if self._hist_bins is not None else 1.0
            return size * bw / 2, y_range * 0.12

    def _on_blob_hover(self, cx: float, cy: float):
        _, kind, xcol, ycol = _PLOT_DEFS[self._plot_combo.currentIndex()]
        rx, ry = self._blob_radii(kind)
        t = np.linspace(0, 2 * np.pi, 64)
        if kind == "scatter":
            self._vb._blob_circle.setData(cx + rx * np.cos(t), cy + ry * np.sin(t))
        else:
            # draw a bracket at the bottom of the histogram
            vr = self._vb.viewRange()
            y0 = vr[1][0]
            xs = [cx - rx, cx - rx, cx + rx, cx + rx]
            ys = [y0 + ry, y0, y0, y0 + ry]
            self._vb._blob_circle.setData(xs, ys)

    def _on_plot_changed(self):
        if self._df is None:
            return
        self._clear_bars()
        self._selected.clear()
        _, kind, xcol, ycol = _PLOT_DEFS[self._plot_combo.currentIndex()]
        if kind == "scatter":
            self._build_scatter(xcol, ycol)
        else:
            self._build_hist(xcol)
        self._refresh_display()
        self._home()

    # ── plot building ─────────────────────────────────────────────────────────

    def _build_scatter(self, xcol: str, ycol: str):
        for sp in self._scat.values():
            sp.setVisible(True)
        self._pi.setLabel("bottom", xcol)
        self._pi.setLabel("left", ycol)
        df = self._df
        for ck in COMBO_KEY_LIST:
            sub = df[df["winner_key"] == ck]
            xs  = sub[xcol].to_numpy(dtype=float)
            ys  = sub[ycol].to_numpy(dtype=float)
            ok  = np.isfinite(xs) & np.isfinite(ys)
            self._scat[ck].setData(xs[ok], ys[ok],
                                   data=sub.index[ok].to_numpy())
        self._hist_bins = None

    def _build_hist(self, xcol: str):
        for sp in self._scat.values():
            sp.setVisible(False)
        self._pi.setLabel("bottom", xcol)
        self._pi.setLabel("left", "count")

        df    = self._df
        all_v = df[xcol].dropna().to_numpy(dtype=float)
        if len(all_v) == 0:
            self._hist_bins = None
            return
        # bin edges fixed from ALL data so axis doesn't shift on subject toggle
        bins  = np.histogram_bin_edges(all_v, bins=_N_BINS)
        self._hist_bins = bins
        width = bins[1] - bins[0]
        cx    = bins[:-1] + width / 2

        subj_ok = (df["subject_id"].isin(self._visible_subjects)
                   if "subject_id" in df.columns else True)
        for ck in COMBO_KEY_LIST:
            sub_v = df[(df["winner_key"] == ck) & subj_ok][xcol].dropna().to_numpy(dtype=float)
            counts, _ = np.histogram(sub_v, bins=bins)
            r, g, b = _hex_rgb(_COLORS[ck])
            bar = pg.BarGraphItem(
                x=cx, height=counts, width=width * 0.85,
                brush=pg.mkBrush(r, g, b, 150),
                pen=pg.mkPen(r, g, b, 220),
            )
            bar.setZValue(5)
            self._pi.addItem(bar)
            self._bars[ck] = bar

    def _clear_bars(self):
        for ck in list(self._bars):
            if self._bars[ck] is not None:
                self._pi.removeItem(self._bars[ck])
                self._bars[ck] = None
        for b in self._sel_bars:
            self._pi.removeItem(b)
        self._sel_bars.clear()

    # ── display refresh ───────────────────────────────────────────────────────

    def _refresh_display(self):
        if self._df is None:
            return
        _, kind, xcol, ycol = _PLOT_DEFS[self._plot_combo.currentIndex()]

        # update displayed data with current combo + subject filters
        df = self._df
        if kind == "scatter":
            subj_ok = (df["subject_id"].isin(self._visible_subjects)
                       if "subject_id" in df.columns else True)
            for ck, sp in self._scat.items():
                if ck not in self._visible:
                    sp.setVisible(False)
                    continue
                sub = df[(df["winner_key"] == ck) & subj_ok]
                xs  = sub[xcol].to_numpy(dtype=float)
                ys  = sub[ycol].to_numpy(dtype=float)
                ok  = np.isfinite(xs) & np.isfinite(ys)
                sp.setData(xs[ok], ys[ok], data=sub.index[ok].to_numpy())
                sp.setVisible(True)
        else:
            self._clear_bars()
            self._build_hist(xcol)
            for ck, bar in self._bars.items():
                if bar is not None:
                    bar.setVisible(ck in self._visible)

        # selection highlight
        if self._selected:
            sel_df  = self._df.loc[sorted(self._selected)]
            vis_sel = sel_df[
                sel_df["winner_key"].isin(self._visible) &
                (sel_df["subject_id"].isin(self._visible_subjects)
                 if "subject_id" in sel_df.columns else True)]
            if kind == "scatter":
                xs = vis_sel[xcol].to_numpy(dtype=float)
                ys = vis_sel[ycol].to_numpy(dtype=float)
                ok = np.isfinite(xs) & np.isfinite(ys)
                self._sel_sp.setData(xs[ok], ys[ok])
                self._sel_sp.setVisible(True)
            else:
                self._sel_sp.setVisible(False)
                self._update_sel_bars(vis_sel, xcol)
        else:
            self._sel_sp.setData([], [])
            self._sel_sp.setVisible(False)
            for b in self._sel_bars:
                self._pi.removeItem(b)
            self._sel_bars.clear()

        n = len(self._selected)
        df = self._df
        self._status_lbl.setText(
            f"{len(df)} ROIs | {n} selected" if n else f"{len(df)} ROIs loaded")
        self._save_btn.setEnabled(n > 0)

    def _update_sel_bars(self, vis_sel: pd.DataFrame, xcol: str):
        for b in self._sel_bars:
            self._pi.removeItem(b)
        self._sel_bars.clear()
        if self._hist_bins is None or vis_sel.empty:
            return
        bins  = self._hist_bins
        width = bins[1] - bins[0]
        cx    = bins[:-1] + width / 2
        sel_v = vis_sel[xcol].dropna().to_numpy(dtype=float)
        if len(sel_v) == 0:
            return
        counts, _ = np.histogram(sel_v, bins=bins)
        nz = counts > 0
        if not nz.any():
            return
        bar = pg.BarGraphItem(
            x=cx[nz], height=counts[nz], width=width * 0.85,
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen("k", width=2),
        )
        bar.setZValue(15)
        self._pi.addItem(bar)
        self._sel_bars.append(bar)

    # ── selection logic ───────────────────────────────────────────────────────

    def _on_lasso(self, pts: list):
        if self._df is None:
            return
        _, kind, xcol, ycol = _PLOT_DEFS[self._plot_combo.currentIndex()]
        vis = self._vis_df()

        if kind == "scatter":
            xs = vis[xcol].to_numpy(dtype=float)
            ys = vis[ycol].to_numpy(dtype=float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            mask = _points_in_polygon(xs[ok], ys[ok], pts)
            hit  = vis.index[ok][mask].tolist()
        else:
            px   = [p[0] for p in pts]
            xlo, xhi = min(px), max(px)
            vals = vis[xcol].to_numpy(dtype=float)
            ok   = np.isfinite(vals)
            mask = ok & (vals >= xlo) & (vals <= xhi)
            hit  = vis.index[mask].tolist()

        self._selected.update(hit)
        self._refresh_display()

    def _on_blob(self, cx: float, cy: float):
        if self._df is None:
            return
        _, kind, xcol, ycol = _PLOT_DEFS[self._plot_combo.currentIndex()]
        rx, ry = self._blob_radii(kind)
        vis = self._vis_df()

        if kind == "scatter":
            xs = vis[xcol].to_numpy(dtype=float)
            ys = vis[ycol].to_numpy(dtype=float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            # normalised ellipse distance ≤ 1
            d2 = ((xs[ok] - cx) / rx) ** 2 + ((ys[ok] - cy) / ry) ** 2
            hit = vis.index[ok][d2 <= 1.0].tolist()
        else:
            vals = vis[xcol].to_numpy(dtype=float)
            ok   = np.isfinite(vals)
            hit  = vis.index[ok & (np.abs(vals - cx) <= rx)].tolist()

        self._selected.update(hit)
        self._refresh_display()

    def _home(self):
        self._pi.enableAutoRange()
        self._pi.autoRange()

    def _clear_selection(self):
        self._selected.clear()
        self._refresh_display()

    def _save(self):
        if self._df is None or not self._selected:
            return
        sel = self._df.loc[sorted(self._selected), ["session_key", "roi_index"]]
        _SAVE_DIR.mkdir(parents=True, exist_ok=True)
        ts  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _SAVE_DIR / f"examples_{ts}.csv"
        sel.to_csv(out, index=False)
        self._status_lbl.setText(f"Saved {len(sel)} ROIs → {out.name}")

    # ── subject overlay ───────────────────────────────────────────────────────

    def _reposition_subject_overlay(self):
        ov = self._subject_overlay
        ov.adjustSize()
        margin = 6
        ov.move(self._pw.width() - ov.width() - margin, margin)
        ov.raise_()

    def eventFilter(self, obj, ev):
        if obj is self._pw and ev.type() == QEvent.Resize:
            self._reposition_subject_overlay()
        return False

    # ── keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, ev: QKeyEvent):
        key = ev.key()
        if key == Qt.Key_H:
            self._home(); ev.accept(); return
        if key == Qt.Key_Z:
            self._clear_combos(); ev.accept(); return
        if key == Qt.Key_A:
            self._select_all_combos(); ev.accept(); return
        if key == Qt.Key_C:
            self._clear_selection(); ev.accept(); return
        if key == Qt.Key_S:
            self._save(); ev.accept(); return
        ck = _KEY_TO_COMBO.get(key)
        if ck:
            self._combo_btns[ck].setChecked(not self._combo_btns[ck].isChecked())
            ev.accept()
            return
        super().keyPressEvent(ev)
