"""PyQt5 + pyqtgraph QC application for dFF baseline inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from .curation import DEFAULT_PATH, derive_category, load_curation, lookup_decision, save_decision
from .data import (
    DEFAULT_DATA_DIR, DEFAULT_PARENT_DIR, aggregate_metrics,
    find_processed_dir, list_sessions, load_session,
)
from .rois import crop_around_mask, get_roi_mask, load_plane_assets, normalize_for_display

# ── visual constants ──────────────────────────────────────────────────────────
BASELINE_COLORS = {
    "short":   "#1f77b4",
    "long":    "#2ca02c",
    "F0trend": "#ff7f0e",
    "F0":      "#d62728",
}
DFF_COLORS = {"short": "#1f77b4", "long": "#2ca02c"}
DFF_DASHED = {"F0trend": (Qt.DashLine, "F0trend"), "F0": (Qt.DotLine, "F0")}
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


# ── trace panel ───────────────────────────────────────────────────────────────

class TracePanel(QWidget):
    """Vertically stacked F + dFF pyqtgraph plots with linked X axis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.f_plot   = pg.PlotWidget(title="Corrected F + baselines")
        self.dff_plot = pg.PlotWidget(title="dFF")
        self.dff_plot.setXLink(self.f_plot)

        for pw in (self.f_plot, self.dff_plot):
            pw.setLabel("bottom", "time (s)")
            pw.addLegend(offset=(5, 5), labelTextSize="8pt")
            pw.showGrid(x=False, y=True, alpha=0.2)
            pw.setDownsampling(auto=True, mode="peak")
            pw.setClipToView(True)

        self.f_plot.setLabel("left", "F")
        self.dff_plot.setLabel("left", "dFF")
        self.f_plot.setMinimumHeight(280)
        self.dff_plot.setMinimumHeight(200)

        layout.addWidget(self.f_plot,   stretch=4)
        layout.addWidget(self.dff_plot, stretch=3)

        self._f_curves:   dict[str, pg.PlotDataItem] = {}
        self._dff_curves: dict[str, pg.PlotDataItem] = {}

    def init_curves(self):
        self.f_plot.clear()
        self.dff_plot.clear()
        self._f_curves = {}
        self._dff_curves = {}

        self._f_curves["F"] = self.f_plot.plot(
            pen=_make_pen("#222", width=1), name="F")

        for name, color in BASELINE_COLORS.items():
            self._f_curves[name] = self.f_plot.plot(
                pen=_make_pen(_pg_color(color), width=2), name=name)

        for name, color in DFF_COLORS.items():
            self._dff_curves[name] = self.dff_plot.plot(
                pen=_make_pen(_pg_color(color), width=1), name=f"dFF ({name})")

        for name, (style, label) in DFF_DASHED.items():
            self._dff_curves[name] = self.dff_plot.plot(
                pen=_make_pen("#000", width=1.5, style=style), name=f"dFF ({label})")

    def update(self, timestamps, F, baselines, dffs):
        self._f_curves["F"].setData(timestamps, F)
        for name, trace in baselines.items():
            if name in self._f_curves:
                self._f_curves[name].setData(timestamps, trace)
        for name, trace in dffs.items():
            if name in self._dff_curves:
                self._dff_curves[name].setData(timestamps, trace)


# ── image panel ───────────────────────────────────────────────────────────────

class ImagePanel(QWidget):
    """FOV + mask overlay with contrast controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # controls row
        ctrl = QHBoxLayout()
        self.mode_btn  = QPushButton("Full FOV")
        self.mask_chk  = QCheckBox("Mask")
        self.mask_chk.setChecked(True)
        ctrl.addWidget(self.mode_btn)
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

        # image view
        self.view = pg.ImageView(view=pg.PlotItem())
        self.view.ui.roiBtn.hide()
        self.view.ui.menuBtn.hide()
        self.view.ui.histogram.hide()
        self.view.setFixedHeight(280)
        self.view.view.setAspectLocked(True)
        layout.addWidget(self.view)

        self._fov_norm: np.ndarray | None = None
        self._mask:     np.ndarray | None = None
        self._zoom_mode = False   # False = zoom crop, True = full FOV

        self.mode_btn.clicked.connect(self._toggle_mode)
        self._lo_slider.valueChanged.connect(self._apply_contrast)
        self._hi_slider.valueChanged.connect(self._apply_contrast)
        self._reset_btn.clicked.connect(self.auto_contrast)
        self.mask_chk.stateChanged.connect(self._redraw)

    # ── public API ────────────────────────────────────────────────────────────

    def set_roi(self, fov_norm: np.ndarray, mask: np.ndarray):
        """fov_norm must already be normalized to [0,1]."""
        self._fov_norm = fov_norm
        self._mask = mask
        self.auto_contrast()
        self._redraw()

    def auto_contrast(self):
        if self._fov_norm is None or self._mask is None:
            return
        fov, mask = self._current_fov_mask()
        px = fov[mask] if mask.any() else fov.ravel()
        lo = float(np.percentile(px, 2))
        hi = float(np.percentile(px, 99))
        span = max(hi - lo, 0.05)
        lo = max(lo - 0.05 * span, 0.0)
        hi = min(hi + 0.05 * span, 1.0)
        # update sliders without firing redraws mid-update
        for sl in (self._lo_slider, self._hi_slider):
            sl.blockSignals(True)
        self._lo_slider.setValue(int(lo * 1000))
        self._hi_slider.setValue(int(hi * 1000))
        for sl in (self._lo_slider, self._hi_slider):
            sl.blockSignals(False)
        self._apply_contrast()

    # ── internal ─────────────────────────────────────────────────────────────

    def _toggle_mode(self):
        self._zoom_mode = not self._zoom_mode
        self.mode_btn.setText("Zoomed" if self._zoom_mode else "Full FOV")
        self.auto_contrast()
        self._redraw()

    def _current_fov_mask(self):
        if self._zoom_mode:
            return self._fov_norm, self._mask
        fov_c, mask_c, _ = crop_around_mask(self._fov_norm, self._mask)
        return fov_c, mask_c

    def _apply_contrast(self):
        lo = self._lo_slider.value() / 1000.0
        hi = max(self._hi_slider.value() / 1000.0, lo + 1e-3)
        self.view.setLevels(lo, hi)

    def _redraw(self):
        if self._fov_norm is None:
            return
        fov, mask = self._current_fov_mask()
        # build RGBA: gray FOV + optional red mask overlay
        h, w = fov.shape
        # pyqtgraph ImageView expects (x, y) or (row, col) with setImage transposing
        # We pass (H, W, 4) uint8 and let pyqtgraph handle it.
        rgba = np.stack([
            (fov * 255).clip(0, 255).astype(np.uint8),
            (fov * 255).clip(0, 255).astype(np.uint8),
            (fov * 255).clip(0, 255).astype(np.uint8),
            np.full((h, w), 255, dtype=np.uint8),
        ], axis=-1)
        if self.mask_chk.isChecked() and mask.any():
            rgba[mask, 0] = MASK_COLOR[0]
            rgba[mask, 1] = MASK_COLOR[1]
            rgba[mask, 2] = MASK_COLOR[2]
            rgba[mask, 3] = 255
        # pyqtgraph ImageView.setImage expects (width, height, channels) — transpose
        self.view.setImage(rgba.transpose(1, 0, 2), autoLevels=False,
                           autoHistogramRange=False)
        self._apply_contrast()


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
            pw = pg.PlotWidget(title=name)
            pw.setFixedHeight(90)
            pw.hideAxis("left")
            pw.getAxis("bottom").setStyle(tickTextOffset=2)
            pw.setMouseEnabled(x=False, y=False)
            pw.setMenuEnabled(False)
            pw.title.setFont(pg.QtGui.QFont("", 8))
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
    def __init__(self, parent_dir: Path, data_dir: Path, output_path: Path):
        super().__init__()
        self.parent_dir  = Path(parent_dir)
        self.data_dir    = Path(data_dir)
        self.output_path = Path(output_path)

        self.setWindowTitle("dFF Baseline QC")
        self.resize(1400, 850)

        # state
        self._sessions: list[Path] = list_sessions(self.parent_dir)
        self._sess_idx  = 0
        self._roi_idx   = 0
        self._session_data = None
        self._curation_df  = load_curation(self.output_path)
        self._agg_metrics  = aggregate_metrics(str(self.parent_dir))

        self._build_ui()
        self._wire_signals()

        pg.setConfigOptions(antialias=False, useOpenGL=True)

        if self._sessions:
            self._load_session(0)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── top bar: session selector + ROI info ──
        top = QHBoxLayout()
        top.addWidget(QLabel("Session:"))
        self.sess_combo = QComboBox()
        self.sess_combo.setMinimumWidth(280)
        for p in self._sessions:
            self.sess_combo.addItem(p.name)
        top.addWidget(self.sess_combo)
        top.addSpacing(16)
        self.roi_label = QLabel("ROI: — / —")
        top.addWidget(self.roi_label)
        self.status_label = QLabel("")
        top.addWidget(self.status_label)
        top.addStretch()
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

        # image
        row = sd.rois.iloc[idx]
        plane_id    = str(row["plane_id"])
        cell_roi_id = int(row["cell_roi_id"])
        proc_dir = find_processed_dir(sd.session_key, str(self.data_dir))
        if proc_dir is not None:
            plane_path = proc_dir / plane_id
            try:
                plane = load_plane_assets(str(plane_path))
                mask  = get_roi_mask(plane, cell_roi_id)
                fov_norm = normalize_for_display(plane.fov)
                self.image_panel.set_roi(fov_norm, mask)
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
            path=self.output_path,
        )
        cat = derive_category(selected, undecided)
        self.status_label.setText(f"Saved → {cat}")

    def _save_and_next(self):
        self._save()
        self._next_roi()

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
        else:
            super().keyPressEvent(event)


# ── entry point ───────────────────────────────────────────────────────────────

def run(parent_dir: Path = DEFAULT_PARENT_DIR,
        data_dir:   Path = DEFAULT_DATA_DIR,
        output:     Path = DEFAULT_PATH):
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(parent_dir=parent_dir, data_dir=data_dir, output_path=output)
    win.show()
    sys.exit(app.exec_())
