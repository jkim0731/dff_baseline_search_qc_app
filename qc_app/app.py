"""PyQt5 + pyqtgraph QC application for dFF baseline inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QMainWindow, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from .curation import DEFAULT_PATH, derive_category, load_curation, lookup_decision, save_decision
from .data import (
    DEFAULT_DATA_DIR, DEFAULT_PARENT_DIR, aggregate_metrics,
    list_sessions, load_session,
)
from .rois import crop_around_mask, get_roi_mask, load_plane_assets, normalize_for_display

# ── visual constants ──────────────────────────────────────────────────────────
BASELINE_COLORS = {
    "short":   "#1f77b4",
    "long":    "#2ca02c",
    "F0trend": "#ff7f0e",
    "F0":      "#d62728",
}
DFF_COLORS = {
    "short":   "#1f77b4",
    "long":    "#2ca02c",
    "F0trend": "#ff7f0e",   # same orange as F baseline
    "F0":      "#d62728",   # same red as F baseline
}
TRACE_KEYS  = ["short", "long", "F0trend", "F0"]   # order matches keys 1-4
METRIC_NAMES = ["noise", "snr", "bleaching", "sustained", "skewness"]
MASK_COLOR = (255, 32, 32, 110)   # RGBA


# ── helpers ───────────────────────────────────────────────────────────────────

def _pg_color(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _make_pen(color, width=1, style=Qt.SolidLine):
    pen = pg.mkPen(color=color, width=width)
    pen.setStyle(style)
    return pen


# ── custom ViewBox: scroll → X zoom only; Shift+scroll → Y zoom only ─────────

class _ShiftYViewBox(pg.ViewBox):
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(ev, axis=1)   # Y only
        else:
            super().wheelEvent(ev, axis=0)   # X only


# ── trace panel ───────────────────────────────────────────────────────────────

class TracePanel(QWidget):
    """F + dFF plots with external legend row and shared trace toggles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── legend row ────────────────────────────────────────────────────────
        legend_row = QHBoxLayout()
        legend_row.setSpacing(4)
        self._legend_btns: dict[str, QPushButton] = {}

        # static "F" label
        f_lbl = QLabel("F")
        f_lbl.setStyleSheet("color:#222; font-size:9pt; font-weight:bold;")
        legend_row.addWidget(f_lbl)

        for i, name in enumerate(TRACE_KEYS, start=1):
            r, g, b = _pg_color(BASELINE_COLORS[name])
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            txt = "white" if luma < 160 else "black"
            btn = QPushButton(f"{i}:{name}")
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedHeight(18)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: rgb({r},{g},{b}); color: {txt};
                    border: none; padding: 1px 7px; font-size: 9pt;
                    border-radius: 3px;
                }}
                QPushButton:!checked {{
                    background: #ccc; color: #999;
                }}
            """)
            btn.toggled.connect(lambda checked, n=name: self._on_toggle(n, checked))
            self._legend_btns[name] = btn
            legend_row.addWidget(btn)

        legend_row.addStretch()
        legend_row.addWidget(QLabel("dFF α:"))
        self._alpha_sl = QSlider(Qt.Horizontal)
        self._alpha_sl.setRange(10, 100)
        self._alpha_sl.setValue(70)
        self._alpha_sl.setFixedWidth(70)
        self._alpha_sl.setFixedHeight(16)
        self._alpha_sl.valueChanged.connect(self._on_alpha_changed)
        legend_row.addWidget(self._alpha_sl)
        self._home_btn = QPushButton("Home")
        self._home_btn.setFixedHeight(18)
        self._home_btn.clicked.connect(self._home)
        legend_row.addWidget(self._home_btn)
        layout.addLayout(legend_row)

        # ── plots ─────────────────────────────────────────────────────────────
        self.f_plot   = pg.PlotWidget(viewBox=_ShiftYViewBox(),
                                      title="Corrected F + baselines")
        self.dff_plot = pg.PlotWidget(viewBox=_ShiftYViewBox(), title="dFF")
        self.dff_plot.setXLink(self.f_plot)

        for pw in (self.f_plot, self.dff_plot):
            pw.setLabel("bottom", "time (s)")
            pw.showGrid(x=False, y=True, alpha=0.2)
            pw.setDownsampling(auto=True, mode="peak")
            pw.setClipToView(True)

        self.f_plot.setLabel("left", "F")
        self.dff_plot.setLabel("left", "dFF")
        self.f_plot.setMinimumHeight(220)
        self.dff_plot.setMinimumHeight(160)

        layout.addWidget(self.f_plot,   stretch=4)
        layout.addWidget(self.dff_plot, stretch=3)

        self._f_curves:   dict[str, pg.PlotDataItem] = {}
        self._dff_curves: dict[str, pg.PlotDataItem] = {}
        self._dff_alpha = 178   # 70 / 100 * 255

    def init_curves(self):
        self.f_plot.clear()
        self.dff_plot.clear()
        self._f_curves = {}
        self._dff_curves = {}

        self._f_curves["F"] = self.f_plot.plot(pen=_make_pen("#222", width=1))

        for name, color in BASELINE_COLORS.items():
            self._f_curves[name] = self.f_plot.plot(
                pen=_make_pen(_pg_color(color), width=2))

        zero = pg.InfiniteLine(pos=0, angle=0,
                               pen=pg.mkPen(color=(180, 180, 180), width=1,
                                            style=Qt.DashLine))
        self.dff_plot.addItem(zero)

        for name, color in DFF_COLORS.items():
            r, g, b = _pg_color(color)
            curve = self.dff_plot.plot(
                pen=_make_pen((r, g, b, self._dff_alpha), width=1))
            self._dff_curves[name] = curve

    def toggle_trace(self, name: str):
        """Toggle a named trace in both plots (driven by button or key press)."""
        btn = self._legend_btns.get(name)
        if btn:
            btn.setChecked(not btn.isChecked())

    def _on_toggle(self, name: str, checked: bool):
        for curves in (self._f_curves, self._dff_curves):
            if name in curves:
                curves[name].setVisible(checked)

    def _on_alpha_changed(self, value: int):
        self._dff_alpha = int(value / 100 * 255)
        for name, curve in self._dff_curves.items():
            r, g, b = _pg_color(DFF_COLORS[name])
            curve.setPen(_make_pen((r, g, b, self._dff_alpha), width=1))

    def _home(self):
        self.f_plot.enableAutoRange()
        self.dff_plot.enableAutoRange()

    def update(self, timestamps, F, baselines, dffs):
        self._f_curves["F"].setData(timestamps, F)
        for name, trace in baselines.items():
            if name in self._f_curves:
                self._f_curves[name].setData(timestamps, trace)
        for name, trace in dffs.items():
            if name in self._dff_curves:
                self._dff_curves[name].setData(timestamps, trace)


# ── bi-state text toggle ─────────────────────────────────────────────────────

class _BiToggleLabel(QLabel):
    """'Option A / Option B' clickable button-style label.

    The active option label is bold black; the inactive one is light gray.
    Parenthesised shortcut hints like '(Z)' are always rendered dark.
    Emits ``toggled`` after each flip (by click or programmatic toggle()).
    """
    toggled = pyqtSignal()

    _HINT_RE = __import__("re").compile(r"^(.*?)(\s*\([^)]+\))$")

    def __init__(self, option_a: str, option_b: str,
                 active: int = 0, parent=None):
        super().__init__(parent)
        self._options = (option_a, option_b)
        self._active  = active % 2
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            "QLabel { border: 1px solid #aaa; border-radius: 3px;"
            "         padding: 2px 7px; background: #ebebeb; }"
            "QLabel:hover { background: #ddd; border-color: #888; }"
        )
        self._refresh()

    def active(self) -> int:
        return self._active

    def setActive(self, idx: int, emit: bool = True):
        idx = idx % 2
        if self._active == idx:
            return
        self._active = idx
        self._refresh()
        if emit:
            self.toggled.emit()

    def toggle(self):
        self.setActive(1 - self._active)

    def mousePressEvent(self, ev):
        self.toggle()
        super().mousePressEvent(ev)

    def _refresh(self):
        parts = []
        for i, opt in enumerate(self._options):
            m = self._HINT_RE.match(opt)
            label, hint = (m.group(1), m.group(2)) if m else (opt, "")
            if i == self._active:
                segment = f"<b style='color:#111'>{label}</b>"
            else:
                segment = f"<span style='color:#bbb'>{label}</span>"
            if hint:
                segment += f"<span style='color:#444'>{hint}</span>"
            parts.append(segment)
        self.setText(" / ".join(parts))


# ── image panel ───────────────────────────────────────────────────────────────

class ImagePanel(QWidget):
    """FOV + mask overlay with contrast, mean/max, zoom/FOV controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # controls row — text toggles + mask checkbox
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        # "Zoom / FOV (Z)": active = 0 → Zoom (cropped), 1 → FOV (full)
        self.zoom_toggle = _BiToggleLabel("Zoom", "FOV (Z)", active=0)
        ctrl.addWidget(self.zoom_toggle)

        # "Mean / Max (A)": active = 0 → Mean, 1 → Max (default)
        self.img_toggle = _BiToggleLabel("Mean", "Max (A)", active=1)
        ctrl.addWidget(self.img_toggle)

        self.mask_chk = QCheckBox("Mask (M)")
        self.mask_chk.setChecked(True)
        ctrl.addWidget(self.mask_chk)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # contrast sliders
        cg = QGroupBox("Contrast")
        cgrid = QGridLayout(cg)
        cgrid.setContentsMargins(4, 4, 4, 4)
        cgrid.setVerticalSpacing(2)
        self._lo_slider = QSlider(Qt.Horizontal)
        self._hi_slider = QSlider(Qt.Horizontal)
        for sl in (self._lo_slider, self._hi_slider):
            sl.setRange(0, 1000)
            sl.setFixedHeight(16)
        self._lo_slider.setValue(0)
        self._hi_slider.setValue(1000)
        self._reset_btn = QPushButton("Auto")
        self._reset_btn.setFixedWidth(44)
        cgrid.addWidget(QLabel("Lo"), 0, 0)
        cgrid.addWidget(self._lo_slider, 0, 1)
        cgrid.addWidget(QLabel("Hi"), 1, 0)
        cgrid.addWidget(self._hi_slider, 1, 1)
        cgrid.addWidget(self._reset_btn, 0, 2, 2, 1)
        layout.addWidget(cg)

        # single RGBA ImageItem
        self.img_plot = pg.PlotWidget()
        self.img_plot.setFixedHeight(220)
        self.img_plot.hideAxis("left")
        self.img_plot.hideAxis("bottom")
        self.img_plot.setAspectLocked(True)
        self.img_plot.setMenuEnabled(False)
        self.img_item = pg.ImageItem()
        self.img_plot.addItem(self.img_item)
        layout.addWidget(self.img_plot)

        self._max_img:  np.ndarray | None = None
        self._mean_img: np.ndarray | None = None
        self._mask:     np.ndarray | None = None

        self.zoom_toggle.toggled.connect(self.auto_contrast)
        self.img_toggle.toggled.connect(self.auto_contrast)
        self._lo_slider.valueChanged.connect(self._redraw)
        self._hi_slider.valueChanged.connect(self._redraw)
        self._reset_btn.clicked.connect(self.auto_contrast)
        self.mask_chk.stateChanged.connect(self._redraw)

    # ── public API ────────────────────────────────────────────────────────────

    def set_roi(self, max_img: np.ndarray, mean_img: np.ndarray, mask: np.ndarray):
        """All images must already be normalized to [0, 1]."""
        self._max_img  = max_img
        self._mean_img = mean_img
        self._mask     = mask
        self.auto_contrast()

    def toggle_zoom(self):
        self.zoom_toggle.toggle()   # emits toggled → auto_contrast

    def toggle_img_mode(self):
        self.img_toggle.toggle()    # emits toggled → auto_contrast

    def auto_contrast(self):
        if self._max_img is None:
            return
        fov, mask = self._current_fov_mask()
        px = fov[mask] if mask.any() else fov.ravel()
        lo = float(np.percentile(px, 2))
        hi = float(np.percentile(px, 99))
        span = max(hi - lo, 0.05)
        lo = max(lo - 0.05 * span, 0.0)
        hi = min(hi + 0.05 * span, 1.0)
        for sl in (self._lo_slider, self._hi_slider):
            sl.blockSignals(True)
        self._lo_slider.setValue(int(lo * 1000))
        self._hi_slider.setValue(int(hi * 1000))
        for sl in (self._lo_slider, self._hi_slider):
            sl.blockSignals(False)
        self._redraw()

    # ── internal ─────────────────────────────────────────────────────────────

    def _current_img(self) -> np.ndarray:
        return self._mean_img if self.img_toggle.active() == 0 else self._max_img

    def _current_fov_mask(self):
        fov = self._current_img()
        if self.zoom_toggle.active() == 1:   # FOV mode
            return fov, self._mask
        fov_c, mask_c, _ = crop_around_mask(fov, self._mask)
        return fov_c, mask_c

    def _redraw(self):
        if self._max_img is None:
            return
        fov, mask = self._current_fov_mask()
        lo = self._lo_slider.value() / 1000.0
        hi = max(self._hi_slider.value() / 1000.0, lo + 1e-3)
        gray8 = (np.clip((fov - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
        h, w = gray8.shape
        rgba = np.stack([gray8, gray8, gray8,
                         np.full((h, w), 255, dtype=np.uint8)], axis=-1)
        if self.mask_chk.isChecked() and mask.any():
            rgba[mask, 0] = MASK_COLOR[0]
            rgba[mask, 1] = MASK_COLOR[1]
            rgba[mask, 2] = MASK_COLOR[2]
            rgba[mask, 3] = 255
        self.img_item.setImage(rgba.transpose(1, 0, 2))
        self.img_plot.autoRange()


# ── metric histograms ─────────────────────────────────────────────────────────

class MetricHistograms(QWidget):
    """Five compact histograms showing per-metric distributions with current ROI marked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._plots: dict[str, pg.PlotWidget] = {}
        self._lines: dict[str, pg.InfiniteLine] = {}
        self._all_metrics: dict[str, np.ndarray] = {}

        for name in METRIC_NAMES:
            pw = pg.PlotWidget()
            pw.setTitle(name, size="8pt")
            pw.setFixedHeight(68)
            pw.hideAxis("left")
            pw.getAxis("bottom").setStyle(tickTextOffset=2)
            pw.setMouseEnabled(x=False, y=False)
            pw.setMenuEnabled(False)
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=2))
            pw.addItem(line)
            self._plots[name] = pw
            self._lines[name] = line
            layout.addWidget(pw)

    def load_all(self, agg_df):
        """Pre-compute histogram bars from aggregate metrics DataFrame."""
        self._all_metrics = {}
        for name in METRIC_NAMES:
            if name in agg_df.columns:
                vals = agg_df[name].to_numpy(dtype=float)
                self._all_metrics[name] = vals[np.isfinite(vals)]
        self._rebuild_bars()

    def _rebuild_bars(self):
        for name, pw in self._plots.items():
            pw.clear()
            # re-add the infinite line after clear
            line = self._lines[name]
            pw.addItem(line)
            vals = self._all_metrics.get(name)
            if vals is None or len(vals) == 0:
                continue
            counts, edges = np.histogram(vals, bins=60)
            bar = pg.BarGraphItem(
                x0=edges[:-1], x1=edges[1:], height=counts,
                brush=pg.mkBrush(136, 136, 136, 180), pen=None)
            pw.addItem(bar)

    def mark_roi(self, metrics_row: dict):
        for name in METRIC_NAMES:
            val = metrics_row.get(name, float("nan"))
            if np.isfinite(float(val)):
                self._lines[name].setValue(float(val))


# ── curation controls ─────────────────────────────────────────────────────────

class CurationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("Label:"))
        self.chk_short   = QCheckBox("short")
        self.chk_long    = QCheckBox("long")
        self.chk_f0trend = QCheckBox("F0trend")
        self.chk_f0      = QCheckBox("F0")
        self.chk_undecided = QCheckBox("undecided")
        for chk in (self.chk_short, self.chk_long, self.chk_f0trend,
                    self.chk_f0, self.chk_undecided):
            layout.addWidget(chk)

        layout.addStretch()
        self.save_btn = QPushButton("Save (S)")
        self.next_btn = QPushButton("Save + Next (Space)")
        self.prev_btn = QPushButton("◀ Prev (J)")
        self.next_roi_btn = QPushButton("Next ▶ (K)")
        for btn in (self.prev_btn, self.next_roi_btn, self.save_btn, self.next_btn):
            layout.addWidget(btn)

    def get_selected(self) -> list[str]:
        out = []
        for key, chk in [("short", self.chk_short), ("long", self.chk_long),
                          ("F0trend", self.chk_f0trend), ("F0", self.chk_f0)]:
            if chk.isChecked():
                out.append(key)
        return out

    def is_undecided(self) -> bool:
        return self.chk_undecided.isChecked()

    def set_state(self, selected: list[str], undecided: bool):
        self.chk_short.setChecked("short" in selected)
        self.chk_long.setChecked("long" in selected)
        self.chk_f0trend.setChecked("F0trend" in selected)
        self.chk_f0.setChecked("F0" in selected)
        self.chk_undecided.setChecked(undecided)

    def clear(self):
        for chk in (self.chk_short, self.chk_long, self.chk_f0trend,
                    self.chk_f0, self.chk_undecided):
            chk.setChecked(False)


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, parent_dir: Path, data_dir: Path, output_path: Path,
                 user: str = ""):
        super().__init__()
        self.parent_dir  = Path(parent_dir)
        self.data_dir    = Path(data_dir)
        self.output_path = Path(output_path)
        self._user       = user

        self.setWindowTitle(f"dFF Baseline QC — {user}" if user else "dFF Baseline QC")
        self.resize(1200, 650)

        # state
        self._sessions: list[Path] = list_sessions(self.parent_dir)
        self._sess_idx  = 0
        self._roi_idx   = 0
        self._session_data = None
        self._curation_df  = load_curation(self.output_path)
        self._agg_metrics  = aggregate_metrics(str(self.parent_dir))
        self._capture_dir: Path | None = None   # resolved lazily; None = use default

        self._build_ui()
        self._wire_signals()

        if self._sessions:
            self._load_session(0)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── top bar ──
        top = QHBoxLayout()
        top.addWidget(QLabel("Session:"))
        self.sess_combo = QComboBox()
        self.sess_combo.setMinimumWidth(280)
        for p in self._sessions:
            self.sess_combo.addItem(p.name)
        top.addWidget(self.sess_combo)
        top.addSpacing(12)
        self.roi_label    = QLabel("ROI: — / —")
        self.plane_label  = QLabel("")
        self.roiid_label  = QLabel("")
        self.status_label = QLabel("")
        for lbl in (self.roi_label, self.plane_label, self.roiid_label,
                    self.status_label):
            top.addWidget(lbl)
            top.addSpacing(8)
        top.addStretch()
        self.capture_btn = QPushButton("Capture (C)")
        self.capture_btn.setFixedHeight(22)
        top.addWidget(self.capture_btn)
        root.addLayout(top)

        # ── body: trace splitter | right column ──
        splitter = QSplitter(Qt.Horizontal)

        self.trace_panel = TracePanel()
        self.trace_panel.init_curves()
        splitter.addWidget(self.trace_panel)

        right = QWidget()
        right.setFixedWidth(360)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        self.image_panel = ImagePanel()
        rl.addWidget(self.image_panel)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        rl.addWidget(line)

        self.metric_hists = MetricHistograms()
        self.metric_hists.load_all(self._agg_metrics)
        rl.addWidget(self.metric_hists)
        rl.addStretch()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, stretch=1)

        # ── bottom: curation ──
        self.curation = CurationPanel()
        root.addWidget(self.curation)

    def _wire_signals(self):
        self.sess_combo.currentIndexChanged.connect(self._on_session_changed)
        self.capture_btn.clicked.connect(self._capture)
        self.curation.prev_btn.clicked.connect(self._prev_roi)
        self.curation.next_roi_btn.clicked.connect(self._next_roi)
        self.curation.save_btn.clicked.connect(self._save)
        self.curation.next_btn.clicked.connect(self._save_and_next)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_session(self, idx: int):
        self._sess_idx = idx
        self._roi_idx  = 0
        self._session_data = load_session(str(self._sessions[idx]))
        self._curation_df  = load_curation(self.output_path)
        self.sess_combo.blockSignals(True)
        self.sess_combo.setCurrentIndex(idx)
        self.sess_combo.blockSignals(False)
        self._refresh_roi()

    def _refresh_roi(self):
        sd = self._session_data
        if sd is None:
            return
        idx = self._roi_idx
        n   = sd.n_rois

        self.roi_label.setText(f"ROI: {idx + 1} / {n}")
        self._show_decision_status()

        ts = sd.timestamps
        F  = np.asarray(sd.F[idx])
        baselines = {k: np.asarray(v[idx]) for k, v in sd.baselines.items()}
        dffs      = {k: np.asarray(v[idx]) for k, v in sd.dffs.items()}
        self.trace_panel.update(ts, F, baselines, dffs)

        # image — load directly from session directory
        row = sd.rois.iloc[idx]
        plane_id    = str(row["plane_id"])
        cell_roi_id = int(row["cell_roi_id"])
        self.plane_label.setText(f"plane: {plane_id}")
        self.roiid_label.setText(f"cell_roi_id: {cell_roi_id}")
        try:
            plane    = load_plane_assets(str(sd.path), plane_id)
            mask     = get_roi_mask(plane, cell_roi_id)
            max_norm  = normalize_for_display(plane.max_img)
            mean_norm = normalize_for_display(plane.mean_img)
            self.image_panel.set_roi(max_norm, mean_norm, mask)
        except Exception as e:
            self.status_label.setText(f"[image error: {e}]")

        # metrics
        metric_row = sd.metrics.iloc[idx].to_dict()
        self.metric_hists.mark_roi(metric_row)

        # curation state
        dec = lookup_decision(self._curation_df, sd.session_key, idx)
        if dec is not None:
            self.curation.set_state(dec["selected_list"], bool(dec["undecided"]))
        else:
            self.curation.clear()

    # ── navigation ────────────────────────────────────────────────────────────

    def _prev_roi(self):
        if self._session_data and self._roi_idx > 0:
            self._roi_idx -= 1
            self._refresh_roi()

    def _next_roi(self):
        if self._session_data and self._roi_idx < self._session_data.n_rois - 1:
            self._roi_idx += 1
            self._refresh_roi()

    def _on_session_changed(self, idx: int):
        self._load_session(idx)

    # ── curation ─────────────────────────────────────────────────────────────

    def _save(self):
        sd = self._session_data
        if sd is None:
            return
        idx         = self._roi_idx
        row         = sd.rois.iloc[idx]
        selected    = self.curation.get_selected()
        undecided   = self.curation.is_undecided()
        self._curation_df = save_decision(
            session_key=sd.session_key,
            roi_index=idx,
            plane_id=str(row["plane_id"]),
            cell_roi_id=int(row["cell_roi_id"]),
            selected=selected,
            undecided=undecided,
            user=self._user,
            path=self.output_path,
        )
        cat = derive_category(selected, undecided)
        self.status_label.setText(f"Saved → {cat}")

    def _save_and_next(self):
        self._save()
        self._next_roi()

    def _resolve_capture_dir(self) -> Path | None:
        """Return a writable captures directory for this session.

        Uses the default (beside the curation CSV) if writable; otherwise
        prompts the user once and caches the choice for the rest of the session.
        """
        if self._capture_dir is not None:
            return self._capture_dir

        default = self.output_path.parent / "captures"
        try:
            default.mkdir(parents=True, exist_ok=True)
            probe = default / ".write_probe"
            probe.touch()
            probe.unlink()
            self._capture_dir = default
            return self._capture_dir
        except (PermissionError, OSError):
            pass

        # Default is read-only — ask the user for an alternative
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Capture folder is read-only — choose a save directory",
            str(Path.home()),
        )
        if not chosen:
            self.status_label.setText("Capture cancelled.")
            return None
        p = Path(chosen)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.status_label.setText(f"Cannot use folder: {exc}")
            return None
        self._capture_dir = p
        return self._capture_dir

    def _capture(self):
        import datetime
        out = self._resolve_capture_dir()
        if out is None:
            return
        sd = self._session_data
        tag = ""
        if sd is not None:
            row = sd.rois.iloc[self._roi_idx]
            tag = (f"_{sd.session_key}_{row['plane_id']}"
                   f"_cell{int(row['cell_roi_id'])}")
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out / f"capture{tag}_{ts}.png"
        self.grab().save(str(path))
        self.status_label.setText(f"Saved: {path}")

    def _show_decision_status(self):
        sd = self._session_data
        if sd is None:
            return
        dec = lookup_decision(self._curation_df, sd.session_key, self._roi_idx)
        if dec:
            self.status_label.setText(f"[{dec['category']}]")
        else:
            self.status_label.setText("")

    # ── keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_J:
            self._prev_roi()
        elif key == Qt.Key_K:
            self._next_roi()
        elif key == Qt.Key_S:
            self._save()
        elif key == Qt.Key_Space:
            self._save_and_next()
        elif key == Qt.Key_1:
            self.trace_panel.toggle_trace("short")
        elif key == Qt.Key_2:
            self.trace_panel.toggle_trace("long")
        elif key == Qt.Key_3:
            self.trace_panel.toggle_trace("F0trend")
        elif key == Qt.Key_4:
            self.trace_panel.toggle_trace("F0")
        elif key == Qt.Key_M:
            self.image_panel.mask_chk.toggle()
        elif key == Qt.Key_Z:
            self.image_panel.toggle_zoom()
        elif key == Qt.Key_A:
            self.image_panel.toggle_img_mode()
        elif key == Qt.Key_C:
            self._capture()
        else:
            super().keyPressEvent(event)


# ── entry point ───────────────────────────────────────────────────────────────

def run(parent_dir: Path | None = None,
        data_dir:   Path = DEFAULT_DATA_DIR,
        output:     Path | None = None):
    pg.setConfigOptions(background="w", foreground="k",
                        antialias=False, useOpenGL=False)
    app = QApplication.instance() or QApplication(sys.argv)

    if parent_dir is None:
        start = str(DEFAULT_PARENT_DIR) if DEFAULT_PARENT_DIR.exists() else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            None, "Select session data folder", start)
        if not chosen:
            sys.exit(0)
        parent_dir = Path(chosen)

    user = ""
    while not user.strip():
        name, ok = QInputDialog.getText(None, "Login", "Enter your name:")
        if not ok:
            sys.exit(0)
        user = name.strip()

    if output is None:
        output = parent_dir / "curation.csv"

    win = MainWindow(parent_dir=parent_dir, data_dir=data_dir,
                     output_path=output, user=user)
    win.show()
    sys.exit(app.exec_())
