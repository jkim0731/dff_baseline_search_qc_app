"""PyQt5 + pyqtgraph QC application for dFF baseline inspection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QSplitter,
    QVBoxLayout, QWidget,
)

from .curation import DEFAULT_PATH, derive_category, load_curation, lookup_decision, save_decision
from .data import (
    DEFAULT_DATA_DIR, DEFAULT_PARENT_DIR, _safe_dff, aggregate_metrics,
    list_sessions, load_session,
)
from .rois import crop_around_mask, get_roi_mask, load_plane_assets
from .runs import DEFAULT_RUNS_DIR, list_run_sessions, load_run_baseline
from .runs_panel import CompareRunsPanel

# ── visual constants ──────────────────────────────────────────────────────────
BASELINE_COLORS = {
    "short":   "#1f77b4",
    "long":    "#2ca02c",
    "F0trend": "#ff7f0e",
    "F0":      "#d62728",
    "run5":    "#9467bd",
    "run6":    "#17becf",
    "run7":    "#bcbd22",
    "run8":    "#e377c2",
}
DFF_COLORS = {
    "short":   "#1f77b4",
    "long":    "#2ca02c",
    "F0trend": "#ff7f0e",
    "F0":      "#d62728",
    "run5":    "#9467bd",
    "run6":    "#17becf",
    "run7":    "#bcbd22",
    "run8":    "#e377c2",
}
TRACE_KEYS = ["short", "long", "F0trend", "F0",
              "run5", "run6", "run7", "run8"]   # order matches keys 1-8
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


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    """1-pixel inner contour: mask pixels that have at least one 4-neighbour outside the mask."""
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1] &
        mask[:-2, 1:-1] & mask[2:, 1:-1] &
        mask[1:-1, :-2] & mask[1:-1, 2:]
    )
    return mask & ~interior


# ── custom ViewBox: scroll → X zoom only; Shift+scroll → Y zoom only ─────────

class _ShiftYViewBox(pg.ViewBox):
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(ev, axis=1)   # Y only
        else:
            super().wheelEvent(ev, axis=0)   # X only


# ── trace panel ───────────────────────────────────────────────────────────────

class TracePanel(QWidget):
    """F + dFF plots with 2-row legend, hover tooltip, color picker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colors: dict[str, str] = dict(BASELINE_COLORS)  # mutable per-instance
        self._color_menu_built = False
        self._available_keys: set[str] = set(TRACE_KEYS)
        self._current_data_keys: set[str] = set()  # keys with non-empty data in curves

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── legend: grid of buttons + controls row ────────────────────────────
        self._legend_btns: dict[str, QPushButton] = {}
        self._legend_ncols = 4   # dynamically adjusted in _relayout_legend()
        legend_area = QVBoxLayout()
        legend_area.setSpacing(1)

        # Controls row — separate so it never gets displaced by button wrapping
        ctrl_row = QHBoxLayout(); ctrl_row.setSpacing(4)
        f_lbl = QLabel("F")
        f_lbl.setStyleSheet("color:#222; font-size:9pt; font-weight:bold;")
        ctrl_row.addWidget(f_lbl)
        ctrl_row.addStretch()
        ctrl_row.addWidget(QLabel("dFF opacity:"))
        self._alpha_sl = QSlider(Qt.Horizontal)
        self._alpha_sl.setRange(10, 100)
        self._alpha_sl.setValue(70)
        self._alpha_sl.setFixedWidth(70)
        self._alpha_sl.setFixedHeight(16)
        self._alpha_sl.valueChanged.connect(self._on_alpha_changed)
        ctrl_row.addWidget(self._alpha_sl)
        self._home_btn = QPushButton("Home")
        self._home_btn.setFixedHeight(18)
        self._home_btn.clicked.connect(self._home)
        ctrl_row.addWidget(self._home_btn)
        legend_area.addLayout(ctrl_row)

        # Trace buttons in a QGridLayout — reflows when compare labels are long
        self._legend_grid_widget = QWidget()
        self._legend_grid = QGridLayout(self._legend_grid_widget)
        self._legend_grid.setContentsMargins(0, 0, 0, 0)
        self._legend_grid.setSpacing(3)
        for i, name in enumerate(TRACE_KEYS, start=1):
            btn = self._make_legend_btn(i, name)
            self._legend_grid.addWidget(btn, (i - 1) // 4, (i - 1) % 4)
        legend_area.addWidget(self._legend_grid_widget)

        layout.addLayout(legend_area)

        # ── plots — single GraphicsLayoutWidget so both repaint in one pass ─────
        self._trace_glw = pg.GraphicsLayoutWidget()
        self.f_plot   = self._trace_glw.addPlot(
            row=0, col=0, viewBox=_ShiftYViewBox(), title="Corrected F + baselines")
        self.dff_plot = self._trace_glw.addPlot(
            row=1, col=0, viewBox=_ShiftYViewBox(), title="dFF")
        self.dff_plot.setXLink(self.f_plot)

        for pi in (self.f_plot, self.dff_plot):
            pi.setLabel("bottom", "time (s)")
            pi.showGrid(x=False, y=True, alpha=0.2)
            pi.setDownsampling(auto=True, mode="peak")
            pi.setClipToView(True)

        self.f_plot.setLabel("left", "F")
        self.dff_plot.setLabel("left", "dFF")

        # Approximate the 4:3 height split used previously
        self._trace_glw.ci.layout.setRowStretchFactor(0, 4)
        self._trace_glw.ci.layout.setRowStretchFactor(1, 3)

        layout.addWidget(self._trace_glw)

        self._f_curves:   dict[str, pg.PlotDataItem] = {}
        self._dff_curves: dict[str, pg.PlotDataItem] = {}
        self._dff_alpha = 178

    # ── legend button factory ─────────────────────────────────────────────────

    def _make_legend_btn(self, i: int, name: str) -> QPushButton:
        btn = QPushButton(f"{i}:{name}")
        btn.setCheckable(True)
        btn.setChecked(True)
        btn.setFixedHeight(18)
        self._apply_btn_style(btn, self._colors.get(name, "#888888"))
        btn.toggled.connect(lambda checked, n=name: self._on_toggle(n, checked))
        btn.setContextMenuPolicy(Qt.CustomContextMenu)
        btn.customContextMenuRequested.connect(lambda pos, n=name: self._pick_color(n))
        self._legend_btns[name] = btn
        return btn

    def _apply_btn_style(self, btn: QPushButton, color_hex: str) -> None:
        r, g, b = _pg_color(color_hex)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        txt = "white" if luma < 160 else "black"
        btn.setStyleSheet(f"""
            QPushButton {{
                background: rgb({r},{g},{b}); color: {txt};
                border: none; padding: 1px 7px; font-size: 9pt;
                border-radius: 3px;
            }}
            QPushButton:!checked {{ background: #ccc; color: #999; }}
        """)

    # ── color picker ──────────────────────────────────────────────────────────

    def _pick_color(self, name: str) -> None:
        from PyQt5.QtGui import QColor
        from PyQt5.QtWidgets import QColorDialog
        current = QColor(self._colors.get(name, "#888888"))
        color = QColorDialog.getColor(current, self, f"Color for: {name}")
        if not color.isValid():
            return
        self._colors[name] = color.name()
        btn = self._legend_btns.get(name)
        if btn is not None:
            self._apply_btn_style(btn, color.name())
        self._apply_curve_pen(name)

    # ── pen management ────────────────────────────────────────────────────────

    def _apply_curve_pen(self, name: str) -> None:
        hex_c = self._colors.get(name, "#888888")
        if name in self._f_curves:
            self._f_curves[name].setPen(_make_pen(_pg_color(hex_c), width=2))
        if name in self._dff_curves:
            r, g, b = _pg_color(hex_c)
            self._dff_curves[name].setPen(_make_pen((r, g, b, self._dff_alpha), width=2))

    # ── plot color submenu ────────────────────────────────────────────────────

    def _build_plot_color_menu(self) -> None:
        for pw in (self.f_plot, self.dff_plot):
            try:
                vb_menu = pw.getViewBox().menu
                if vb_menu is None:
                    continue
                vb_menu.addSeparator()
                color_sub = vb_menu.addMenu("Set trace color…")
                for name in TRACE_KEYS:
                    act = color_sub.addAction(name)
                    act.triggered.connect(lambda _, n=name: self._pick_color(n))
            except Exception:
                pass

    # ── curve initialisation ──────────────────────────────────────────────────

    def init_curves(self):
        self.f_plot.clear()
        self.dff_plot.clear()
        self._f_curves = {}
        self._dff_curves = {}

        # Use plot() so each item is a PlotDataItem that inherits the PlotItem's
        # autoDownsample and clipToView settings (set above). Width=2 is fast
        # because useOpenGL=True offloads stroke rendering to the GPU.
        self._f_curves["F"] = self.f_plot.plot(pen=_make_pen("#222", width=1))

        for name in TRACE_KEYS:
            c = self.f_plot.plot(
                pen=_make_pen(_pg_color(self._colors.get(name, "#888")), width=2),
            )
            self._f_curves[name] = c

        zero = pg.InfiniteLine(pos=0, angle=0,
                               pen=pg.mkPen(color=(180, 180, 180), width=1,
                                            style=Qt.DashLine))
        self.dff_plot.addItem(zero)

        for name in TRACE_KEYS:
            r, g, b = _pg_color(self._colors.get(name, "#888"))
            c = self.dff_plot.plot(
                pen=_make_pen((r, g, b, self._dff_alpha), width=2),
            )
            self._dff_curves[name] = c

        # Color submenu in plot context menu (only added once)
        if not self._color_menu_built:
            self._build_plot_color_menu()
            self._color_menu_built = True

    def toggle_trace(self, name: str):
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
            r, g, b = _pg_color(self._colors.get(name, "#888"))
            curve.setPen(_make_pen((r, g, b, self._dff_alpha), width=2))

    def _home(self):
        self.f_plot.enableAutoRange()
        self.dff_plot.enableAutoRange()

    def set_available_keys(self, keys: list[str]):
        self._available_keys = set(keys)
        empty = np.array([])
        for name, btn in self._legend_btns.items():
            available = name in self._available_keys
            btn.setVisible(available)
            if not available:
                if name in self._f_curves:
                    self._f_curves[name].setData(empty, empty)
                if name in self._dff_curves:
                    self._dff_curves[name].setData(empty, empty)
        # Reset tracking so update() re-evaluates transitions on next call
        self._current_data_keys = set()

    def set_slot_label(self, slot_key: str, label: str | None):
        btn = self._legend_btns.get(slot_key)
        if btn is None:
            return
        i = list(self._legend_btns).index(slot_key) + 1
        btn.setText(f"{i}:{slot_key}" if label is None else f"{i}:{label}")
        self._relayout_legend()

    def _relayout_legend(self) -> None:
        """Switch between 4-col and 2-col grid based on longest button label."""
        max_len = max(len(btn.text()) for btn in self._legend_btns.values())
        ncols = 4 if max_len <= 12 else 2
        if ncols == self._legend_ncols:
            return
        self._legend_ncols = ncols
        for i, btn in enumerate(self._legend_btns.values()):
            self._legend_grid.addWidget(btn, i // ncols, i % ncols)
        # After moving widgets Qt may have stale effective-visibility state;
        # re-apply the current visibility for every button unconditionally.
        for name, btn in self._legend_btns.items():
            btn.setVisible(name in self._available_keys)

    def update(self, timestamps, F, baselines, dffs):
        empty = np.array([])
        self._f_curves["F"].setData(timestamps, F)
        new_data_keys = set(baselines) | set(dffs)
        for name in TRACE_KEYS:
            had_data = name in self._current_data_keys
            has_f    = name in baselines
            has_dff  = name in dffs
            if has_f and name in self._f_curves:
                self._f_curves[name].setData(timestamps, baselines[name])
            elif had_data and name in self._f_curves:
                self._f_curves[name].setData(empty, empty)
            if has_dff and name in self._dff_curves:
                self._dff_curves[name].setData(timestamps, dffs[name])
            elif had_data and name in self._dff_curves:
                self._dff_curves[name].setData(empty, empty)
            # Button visibility is managed solely by set_available_keys; do not
            # hide buttons here based on data presence — that caused slots 5/7
            # to stay invisible after assignment due to stale Qt visibility state.
        self._current_data_keys = new_data_keys


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
            contour = _mask_contour(mask)
            rgba[contour, 0] = MASK_COLOR[0]
            rgba[contour, 1] = MASK_COLOR[1]
            rgba[contour, 2] = MASK_COLOR[2]
            rgba[contour, 3] = 255
        self.img_item.setImage(rgba.transpose(1, 0, 2))
        self.img_plot.autoRange()


# ── metric histograms ─────────────────────────────────────────────────────────

class MetricHistograms(QWidget):
    """Compact histograms — all in one GraphicsLayoutWidget so a single repaint
    covers every histogram when the marker line moves."""

    _ROW_H = 82  # pixels per histogram row

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._plots: dict[str, pg.PlotItem] = {}
        self._lines: dict[str, pg.InfiniteLine] = {}
        self._all_metrics: dict[str, np.ndarray] = {}

        glw = pg.GraphicsLayoutWidget()
        glw.setFixedHeight(len(METRIC_NAMES) * self._ROW_H)
        layout.addWidget(glw)

        for i, name in enumerate(METRIC_NAMES):
            pi = glw.addPlot(row=i, col=0)
            pi.setTitle(name, size="8pt")
            pi.hideAxis("left")
            pi.getAxis("bottom").setStyle(tickTextOffset=2)
            pi.setMouseEnabled(x=False, y=False)
            pi.setMenuEnabled(False)
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=1))
            pi.addItem(line)
            self._plots[name] = pi
            self._lines[name] = line

    def load_all(self, agg_df):
        """Pre-compute histogram bars from aggregate metrics DataFrame."""
        self._all_metrics = {}
        for name in METRIC_NAMES:
            if name in agg_df.columns:
                vals = agg_df[name].to_numpy(dtype=float)
                self._all_metrics[name] = vals[np.isfinite(vals)]
        self._rebuild_bars()

    def _rebuild_bars(self):
        for name, pi in self._plots.items():
            pi.clear()
            pi.addItem(self._lines[name])
            vals = self._all_metrics.get(name)
            if vals is None or len(vals) == 0:
                continue
            counts, edges = np.histogram(vals, bins=60)
            pi.addItem(pg.BarGraphItem(
                x0=edges[:-1], x1=edges[1:], height=counts,
                brush=pg.mkBrush(136, 136, 136, 180), pen=None))

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

    def set_available_keys(self, keys: list[str]):
        """Show only checkboxes for baseline keys present in this session."""
        for key, chk in [("short", self.chk_short), ("long", self.chk_long),
                          ("F0trend", self.chk_f0trend), ("F0", self.chk_f0)]:
            available = key in keys
            chk.setVisible(available)
            if not available:
                chk.setChecked(False)

    def clear(self):
        for chk in (self.chk_short, self.chk_long, self.chk_f0trend,
                    self.chk_f0, self.chk_undecided):
            chk.setChecked(False)


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, parent_dir: Path, data_dir: Path, output_path: Path,
                 user: str = "",
                 runs_dirs: list[Path] | Path | None = None):
        super().__init__()
        self.parent_dir  = Path(parent_dir)
        self.data_dir    = Path(data_dir)
        self.output_path = Path(output_path)
        self._user       = user
        # Normalize to list[Path]; accept None / single Path / list for flexibility.
        if runs_dirs is None:
            self._runs_dirs: list[Path] = []
        elif isinstance(runs_dirs, (str, Path)):
            self._runs_dirs = [Path(runs_dirs)]
        else:
            self._runs_dirs = [Path(d) for d in runs_dirs]

        self.setWindowTitle(f"dFF Baseline QC — {user}" if user else "dFF Baseline QC")
        # Width is resizable, height is fixed so the curation row at the bottom
        # is always visible (compare dock pops out as its own floating window).
        self.resize(1200, 720)

        # state
        self._all_sessions: list[Path] = list_sessions(self.parent_dir)
        # When compare-mode slot overrides are active, _sessions is restricted
        # to the intersection of session keys present across the selected runs.
        self._sessions: list[Path] = list(self._all_sessions)
        self._sess_idx  = 0
        self._roi_idx   = 0
        self._session_data = None
        self._curation_df  = load_curation(self.output_path)
        self._agg_metrics  = aggregate_metrics(str(self.parent_dir))
        self._capture_dir: Path | None = None   # resolved lazily; None = use default
        # slot_key -> (run_dir: Path, kind: str, label: str). Empty ⇒ legacy.
        self._slot_overrides: dict[str, tuple[Path, str, str]] = {}

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
        if self._runs_dirs:
            self.compare_runs_btn = QPushButton("Compare mode (R)")
            self.compare_runs_btn.setCheckable(True)
            self.compare_runs_btn.setFixedHeight(22)
            self.compare_runs_btn.setToolTip(
                "Toggle the persistent compare panel "
                f"(reads runs from {len(self._runs_dirs)} source(s); add more from inside the panel)"
            )
            top.addWidget(self.compare_runs_btn)
        else:
            self.compare_runs_btn = None
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

        # ── compare-mode dock (only when runs_dir is set) ──
        # Floats by default so toggling it on/off never reflows or hides any
        # part of the main window. The user can still drag-dock it on demand.
        self.compare_panel = None
        self.compare_dock = None
        if self._runs_dirs:
            self.compare_panel = CompareRunsPanel(
                self._runs_dirs, current=self._slot_overrides
            )
            self.compare_dock = QDockWidget("Compare mode", self)
            self.compare_dock.setObjectName("compare_dock")
            self.compare_dock.setWidget(self.compare_panel)
            self.compare_dock.setAllowedAreas(
                Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea
            )
            self.addDockWidget(Qt.BottomDockWidgetArea, self.compare_dock)
            self.compare_dock.setFloating(True)
            # Remove Qt.Tool so the floating dock can go behind the main window.
            self.compare_dock.setWindowFlags(
                Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
            )
            self.compare_dock.topLevelChanged.connect(self._on_dock_toplevel_changed)
            # Place beside the main window with a sensible default size
            self.compare_dock.resize(1100, 460)
            self.compare_dock.hide()   # hidden by default — toggle via "Compare mode (R)"

    def _wire_signals(self):
        self.sess_combo.currentIndexChanged.connect(self._on_session_changed)
        self.capture_btn.clicked.connect(self._capture)
        self.curation.prev_btn.clicked.connect(self._prev_roi)
        self.curation.next_roi_btn.clicked.connect(self._next_roi)
        self.curation.save_btn.clicked.connect(self._save)
        self.curation.next_btn.clicked.connect(self._save_and_next)
        if self.compare_runs_btn is not None and self.compare_dock is not None:
            self.compare_runs_btn.toggled.connect(self._set_compare_mode)
            self.compare_dock.visibilityChanged.connect(
                lambda v: self.compare_runs_btn.setChecked(v)
            )
            self.compare_panel.selectionsChanged.connect(self._on_compare_selections)

    def _on_dock_toplevel_changed(self, floating: bool):
        if floating:
            # Re-apply Qt.Window so the dock behaves as a normal window
            # (can go behind the main window) rather than a always-on-top tool.
            self.compare_dock.setWindowFlags(
                Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
            )
            self.compare_dock.show()

    def _set_compare_mode(self, on: bool):
        if self.compare_dock is None:
            return
        self.compare_dock.setVisible(on)
        if on and self.compare_dock.isFloating():
            self.compare_dock.raise_()

    def _on_compare_selections(self, selections: dict):
        self._slot_overrides = selections
        for slot_key in TRACE_KEYS:
            sel = self._slot_overrides.get(slot_key)
            self.trace_panel.set_slot_label(
                slot_key, None if sel is None else sel[2]
            )
        already_refreshed = self._apply_session_intersection_filter()
        if not already_refreshed:
            self._refresh_roi()

    def _apply_session_intersection_filter(self) -> bool:
        """Restrict sess_combo to sessions present in every assigned run.

        Returns True if it already called _refresh_roi (via _load_session) so
        the caller can avoid a redundant second refresh.
        """
        if not self._slot_overrides:
            new_sessions = list(self._all_sessions)
            note = ""
        else:
            run_dirs = {ovr[0] for ovr in self._slot_overrides.values()}
            per_run = [set(list_run_sessions(rd)) for rd in run_dirs]
            shared = set.intersection(*per_run) if per_run else set()
            new_sessions = [p for p in self._all_sessions if p.name in shared]
            note = (f"  ·  compare-mode session filter: "
                    f"{len(new_sessions)}/{len(self._all_sessions)} sessions "
                    f"shared across {len(run_dirs)} run(s)")

        # No-op fast path — session list unchanged, no refresh needed here
        if [p.name for p in new_sessions] == [p.name for p in self._sessions]:
            if note:
                self.status_label.setText(note.lstrip("  ·  "))
            return False

        cur_name = self._sessions[self._sess_idx].name if self._sessions else None
        self._sessions = new_sessions

        self.sess_combo.blockSignals(True)
        self.sess_combo.clear()
        for p in self._sessions:
            self.sess_combo.addItem(p.name)
        self.sess_combo.blockSignals(False)

        if not self._sessions:
            self.status_label.setText(
                "No sessions are shared across the selected runs — "
                "uncheck or reassign slots."
            )
            return False
        # Try to keep the user on the same session; otherwise jump to first.
        # _load_session calls _refresh_roi internally.
        names = [p.name for p in self._sessions]
        if cur_name in names:
            self._load_session(names.index(cur_name))
        else:
            self._load_session(0)
        if note:
            self.status_label.setText(note.lstrip("  ·  "))
        return True

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_session(self, idx: int):
        self._sess_idx = idx
        self._roi_idx  = 0
        self._session_data = load_session(str(self._sessions[idx]))
        self._curation_df  = load_curation(self.output_path)
        self.sess_combo.blockSignals(True)
        self.sess_combo.setCurrentIndex(idx)
        self.sess_combo.blockSignals(False)
        # In runs-comparison mode, all 8 slots are always available (legacy or overridden).
        keys = (TRACE_KEYS
                if self._runs_dirs
                else list(self._session_data.baselines.keys()))
        self.trace_panel.set_available_keys(keys)
        self.curation.set_available_keys(keys)
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
        baselines, dffs = self._build_slot_traces(sd, idx, F)
        self.trace_panel.update(ts, F, baselines, dffs)

        # image — load directly from session directory
        row = sd.rois.iloc[idx]
        plane_id    = str(row["plane_id"])
        cell_roi_id = int(row["cell_roi_id"])
        self.plane_label.setText(f"plane: {plane_id}")
        self.roiid_label.setText(f"cell_roi_id: {cell_roi_id}")
        try:
            plane = load_plane_assets(str(sd.path), plane_id)
            mask  = get_roi_mask(str(sd.path), plane_id, cell_roi_id)
            self.image_panel.set_roi(plane.max_norm, plane.mean_norm, mask)
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

    def _build_slot_traces(self, sd, idx: int, F: np.ndarray):
        """Per-slot (baseline, dff) for the current ROI.

        For each of the 8 slot keys, prefer an override mapped to a run folder;
        otherwise fall back to ``sd.baselines[slot_key]`` (legacy, may be absent).
        """
        baselines: dict = {}
        dffs: dict = {}
        for slot_key in TRACE_KEYS:
            override = self._slot_overrides.get(slot_key)
            if override is not None:
                run_dir, kind, _label = override
                try:
                    arr = load_run_baseline(str(run_dir), sd.session_key, kind)
                except FileNotFoundError:
                    self.status_label.setText(
                        f"[{slot_key}] missing in {Path(run_dir).name} for {sd.session_key}"
                    )
                    continue
                b = np.asarray(arr[idx])
                baselines[slot_key] = b
                dffs[slot_key]      = _safe_dff(F, b)
                continue
            # legacy fallback
            if slot_key in sd.baselines:
                baselines[slot_key] = np.asarray(sd.baselines[slot_key][idx])
                dffs[slot_key]      = np.asarray(sd.dffs[slot_key][idx])
        return baselines, dffs

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
        elif key == Qt.Key_5:
            self.trace_panel.toggle_trace("run5")
        elif key == Qt.Key_6:
            self.trace_panel.toggle_trace("run6")
        elif key == Qt.Key_7:
            self.trace_panel.toggle_trace("run7")
        elif key == Qt.Key_8:
            self.trace_panel.toggle_trace("run8")
        elif key == Qt.Key_M:
            self.image_panel.mask_chk.toggle()
        elif key == Qt.Key_Z:
            self.image_panel.toggle_zoom()
        elif key == Qt.Key_A:
            self.image_panel.toggle_img_mode()
        elif key == Qt.Key_C:
            self._capture()
        elif key == Qt.Key_R and self.compare_runs_btn is not None:
            self.compare_runs_btn.toggle()
        else:
            super().keyPressEvent(event)


# ── entry point ───────────────────────────────────────────────────────────────

def _pick_dir(title: str, start: Path) -> Path | None:
    start_str = str(start) if start.exists() else str(Path.home())
    chosen = QFileDialog.getExistingDirectory(None, title, start_str)
    return Path(chosen) if chosen else None


def run(parent_dir: Path | None = None,
        data_dir:   Path = DEFAULT_DATA_DIR,
        output:     Path | None = None,
        runs_dirs:  list[Path] | None = None):
    pg.setConfigOptions(background="w", foreground="k",
                        antialias=False, useOpenGL=True)
    app = QApplication.instance() or QApplication(sys.argv)

    if parent_dir is None:
        start = DEFAULT_RUNS_DIR if DEFAULT_RUNS_DIR.exists() else DEFAULT_PARENT_DIR
        msg = None
        while True:
            title = (
                "Select inputs folder (contains session subfolders with F_all_array.npy)"
                if msg is None else
                f"⚠ {msg} — pick again, or Cancel to quit"
            )
            parent_dir = _pick_dir(title, start)
            if parent_dir is None:
                sys.exit(0)
            sessions = list_sessions(parent_dir)
            if sessions:
                break
            # Diagnose why nothing was found
            subdirs = [p for p in parent_dir.iterdir() if p.is_dir()] if parent_dir.exists() else []
            if not subdirs:
                msg = f"'{parent_dir.name}' is empty"
            elif any((d / "recipe.json").exists() for d in subdirs):
                msg = f"'{parent_dir.name}' looks like a runs folder — pick the inputs subfolder inside it"
            else:
                msg = f"No session subfolders with F_all_array.npy found in '{parent_dir.name}'"
            start = parent_dir

    # The parent of the inputs folder is treated as the runs root.
    if runs_dirs is None:
        runs_dirs = [parent_dir.parent]

    user = ""
    while not user.strip():
        name, ok = QInputDialog.getText(None, "Login", "Enter your name:")
        if not ok:
            sys.exit(0)
        user = name.strip()

    if output is None:
        # parent_dir may be read-only on CodeOcean — fall back to $HOME if so.
        try:
            (parent_dir / ".write_probe").touch()
            (parent_dir / ".write_probe").unlink()
            output = parent_dir / "curation.csv"
        except (PermissionError, OSError):
            output = Path.home() / "dff_baseline_qc_curation.csv"

    win = MainWindow(parent_dir=parent_dir, data_dir=data_dir,
                     output_path=output, user=user, runs_dirs=runs_dirs)
    if runs_dirs and getattr(win, "compare_runs_btn", None) is not None:
        # Open the compare dock by default so the picker is immediately visible.
        win.compare_runs_btn.setChecked(True)
    win.show()
    sys.exit(app.exec_())
