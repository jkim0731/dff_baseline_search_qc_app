"""PyQt5 + pyqtgraph QC app: verify binit0 (c_pos,c_neg) noise-criterion matches."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QPalette
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QRadioButton,
    QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from base_qc_app.rois import crop_around_mask, get_roi_mask, load_plane_assets

from .curation import (DEFAULT_PATH, FLAG_COLS, FLAGS,
                        load_curation, lookup_decision, save_decision)
from .data import (
    COMBO_KEY, COMBO_KEY_LIST, COMBO_KEYS, COMBO_LABEL, COMBOS,
    KEY_COMBO, METRIC_DISPLAY, METRIC_LOG, METRIC_NAMES, PARAM_NAMES,
    TARGET_COEF, TRACE_KEYS,
    _safe_dff, aggregate_metrics, compute_model_components, compute_noise_bar,
    discover_combo_runs, list_sessions, load_session,
)

# ── colors ────────────────────────────────────────────────────────────────────
TRACE_COLORS = {
    "short": "#1f77b4",   # blue
    "long":  "#2ca02c",   # green
    "c23":   "#ff7f0e",   # orange
    "c24":   "#d62728",   # red
    "c25":   "#9467bd",   # purple
    "c33":   "#17becf",   # teal
    "c34":   "#bcbd22",   # olive
    "c35":   "#e377c2",   # pink
    "c44":   "#7f7f7f",   # gray
    "c45":   "#8c564b",   # brown
}
TRACE_LABELS = {
    "short": "short",
    "long":  "long",
    **{COMBO_KEY[c]: COMBO_LABEL[c] for c in COMBOS},
}
MASK_COLOR   = (255, 32, 32, 110)

# Component breakdown colors & line styles (F plot only)
COMP_STYLES: dict[str, tuple] = {
    "b_inf":    ("#555555", Qt.DashLine),
    "b_slow":   ("#1565C0", Qt.SolidLine),
    "b_fast":   ("#C62828", Qt.SolidLine),
    "b_bright": ("#2E7D32", Qt.SolidLine),
}


def _pg_color(hex_c: str) -> tuple:
    h = hex_c.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _make_pen(color, width=1, style=Qt.SolidLine):
    pen = pg.mkPen(color=color, width=width)
    pen.setStyle(style)
    return pen


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] &
        mask[1:-1, :-2]  & mask[1:-1, 2:]
    )
    return mask & ~interior


# ── scroll-to-zoom ViewBox ────────────────────────────────────────────────────

class _ShiftYViewBox(pg.ViewBox):
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(ev, axis=1)
        else:
            super().wheelEvent(ev, axis=0)


# ── trace panel (10 traces) ───────────────────────────────────────────────────

class TracePanel(QWidget):
    """F + baselines / dFF plots for the 10 fixed trace keys."""

    baseline_mode_changed = pyqtSignal()   # emitted when IRLS ↔ LOWESS toggled

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colors = dict(TRACE_COLORS)
        self._current_data_keys: set = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # ── legend ────────────────────────────────────────────────────────────
        legend_area = QVBoxLayout(); legend_area.setSpacing(2)

        # top ctrl row: IRLS/LOWESS (disabled) | Baseline/Components | stretch | Clear | All | Home
        ctrl = QHBoxLayout(); ctrl.setSpacing(4)
        self._baseline_mode = _BiToggle("IRLS", "LOWESS", active=0)
        self._baseline_mode.setEnabled(False)
        self._baseline_mode.toggled.connect(self.baseline_mode_changed.emit)
        ctrl.addWidget(self._baseline_mode)
        self._comp_toggle = _BiToggle("Baseline", "Components (V)", active=0)
        self._comp_toggle.toggled.connect(self._apply_comp_mode)
        ctrl.addWidget(self._comp_toggle)
        ctrl.addStretch()
        self._clear_btn = QPushButton("Clear (Z)")
        self._clear_btn.setFixedHeight(18)
        self._clear_btn.clicked.connect(self.clear_traces)
        ctrl.addWidget(self._clear_btn)
        self._all_btn = QPushButton("All (A)")
        self._all_btn.setFixedHeight(18)
        self._all_btn.clicked.connect(self.select_all_traces)
        ctrl.addWidget(self._all_btn)
        self._home_btn = QPushButton("Home (H)")
        self._home_btn.setFixedHeight(18)
        self._home_btn.clicked.connect(self._home)
        ctrl.addWidget(self._home_btn)
        legend_area.addLayout(ctrl)

        # Single row of 10 compact buttons (key 0 = last trace c45)
        self._legend_btns: dict[str, QPushButton] = {}
        btn_row = QHBoxLayout(); btn_row.setSpacing(2); btn_row.setContentsMargins(0,0,0,0)
        for i, key in enumerate(TRACE_KEYS):
            shortcut = "0" if i == len(TRACE_KEYS) - 1 else str(i + 1)
            label    = TRACE_LABELS[key]
            btn      = QPushButton(f"{shortcut}:{label}")
            btn.setCheckable(True); btn.setChecked(True)
            btn.setFixedHeight(18)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._apply_btn_style(btn, self._colors[key])
            btn.toggled.connect(lambda checked, k=key: self._on_toggle(k, checked))
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, k=key: self._pick_color(k))
            self._legend_btns[key] = btn
            btn_row.addWidget(btn)
        legend_area.addLayout(btn_row)

        layout.addLayout(legend_area)

        # ── plots ─────────────────────────────────────────────────────────────
        self._trace_glw = pg.GraphicsLayoutWidget()
        self.f_plot   = self._trace_glw.addPlot(row=0, col=0, viewBox=_ShiftYViewBox(),
                                                 title="Corrected F + baselines")
        self.dff_plot = self._trace_glw.addPlot(row=1, col=0, viewBox=_ShiftYViewBox(),
                                                 title="dFF")
        self.dff_plot.setXLink(self.f_plot)
        for pi in (self.f_plot, self.dff_plot):
            pi.setLabel("bottom", "time (s)")
            pi.showGrid(x=False, y=True, alpha=0.2)
            pi.setDownsampling(auto=True, mode="peak")
            pi.setClipToView(True)
        self.f_plot.setLabel("left", "F")
        self.dff_plot.setLabel("left", "dFF")
        self._trace_glw.ci.layout.setRowStretchFactor(0, 4)
        self._trace_glw.ci.layout.setRowStretchFactor(1, 3)
        layout.addWidget(self._trace_glw)

        param_row = QHBoxLayout(); param_row.setContentsMargins(0, 0, 0, 0); param_row.setSpacing(0)
        self._param_lbl = QLabel("")
        self._param_lbl.setStyleSheet(
            "font-size: 8pt; color: #333; font-family: monospace; padding: 1px 4px;")
        self._noise_lbl = QLabel("")
        self._noise_lbl.setStyleSheet(
            "font-size: 8pt; color: #555; font-family: monospace; padding: 1px 4px;")
        self._noise_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        param_row.addWidget(self._param_lbl, stretch=1)
        param_row.addWidget(self._noise_lbl)
        layout.addLayout(param_row)

        self._f_curves:   dict = {}
        self._dff_curves: dict = {}
        self._comp_curves: dict = {}
        self._comp_key:  str | None = None
        self._comp_ts:   object     = None
        self._comp_data: dict | None = None
        self._color_menu_built = False

    @property
    def baseline_mode(self) -> str:
        """'irls' (F0trend) or 'lowess' (F0)."""
        return "lowess" if self._baseline_mode.active() == 1 else "irls"

    def _apply_btn_style(self, btn, color_hex):
        r, g, b = _pg_color(color_hex)
        luma = 0.299*r + 0.587*g + 0.114*b
        txt  = "white" if luma < 160 else "black"
        btn.setStyleSheet(f"""
            QPushButton {{
                background: rgb({r},{g},{b}); color: {txt};
                border: none; padding: 1px 7px; font-size: 9pt; border-radius: 3px;
            }}
            QPushButton:!checked {{ background: #ccc; color: #999; }}
        """)

    def _pick_color(self, key):
        from PyQt5.QtGui import QColor
        from PyQt5.QtWidgets import QColorDialog
        color = QColorDialog.getColor(QColor(self._colors[key]), self, f"Color: {key}")
        if not color.isValid():
            return
        self._colors[key] = color.name()
        self._apply_btn_style(self._legend_btns[key], color.name())
        self._apply_curve_pen(key)

    def _apply_curve_pen(self, key):
        hex_c = self._colors[key]
        pen = _make_pen(_pg_color(hex_c), width=2)
        if key in self._f_curves:
            self._f_curves[key].setPen(pen)
        if key in self._dff_curves:
            self._dff_curves[key].setPen(pen)

    def _build_plot_color_menu(self):
        for pw in (self.f_plot, self.dff_plot):
            try:
                vb_menu = pw.getViewBox().menu
                if vb_menu is None:
                    continue
                vb_menu.addSeparator()
                sub = vb_menu.addMenu("Set trace color…")
                for key in TRACE_KEYS:
                    act = sub.addAction(TRACE_LABELS[key])
                    act.triggered.connect(lambda _, k=key: self._pick_color(k))
            except Exception:
                pass

    def init_curves(self):
        self.f_plot.clear(); self.dff_plot.clear()
        self._f_curves = {}; self._dff_curves = {}; self._comp_curves = {}

        self._f_curves["F"] = self.f_plot.plot(pen=_make_pen("#222", width=1))
        for key in TRACE_KEYS:
            self._f_curves[key] = self.f_plot.plot(
                pen=_make_pen(_pg_color(self._colors[key]), width=2))
        self.dff_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color=(180,180,180), width=1, style=Qt.DashLine)))
        for key in TRACE_KEYS:
            self._dff_curves[key] = self.dff_plot.plot(
                pen=_make_pen(_pg_color(self._colors[key]), width=2))
        self._comp_legend = pg.LegendItem(
            offset=(-10, 10), colCount=len(COMP_STYLES),
            labelTextSize="8pt",
        )
        self._comp_legend.setParentItem(self.f_plot.vb)
        self._comp_legend.setVisible(False)
        for name, (color, style) in COMP_STYLES.items():
            c = self.f_plot.plot(pen=_make_pen(_pg_color(color), width=2, style=style))
            c.setVisible(False)
            self._comp_curves[name] = c
            self._comp_legend.addItem(c, name)
            _, label = self._comp_legend.items[-1]
            label.setText(name, color=color)
        if not self._color_menu_built:
            self._build_plot_color_menu()
            self._color_menu_built = True

    def toggle_trace(self, key):
        btn = self._legend_btns.get(key)
        if btn:
            btn.setChecked(not btn.isChecked())

    def _on_toggle(self, key, checked):
        if key in self._dff_curves:
            self._dff_curves[key].setVisible(checked)
        if key in self._f_curves:
            comp_on = self._comp_toggle.active() == 1
            hide_f = comp_on and key == self._comp_key
            self._f_curves[key].setVisible(checked and not hide_f)

    def _home(self):
        self.f_plot.enableAutoRange()
        self.dff_plot.enableAutoRange()

    def toggle_comp_mode(self):
        self._comp_toggle.toggle()

    def set_component_data(
        self,
        key: str | None,
        ts: np.ndarray,
        comp_dict: dict | None,
    ):
        self._comp_key  = key
        self._comp_ts   = ts
        self._comp_data = comp_dict
        if comp_dict is not None and ts is not None:
            for name, curve in self._comp_curves.items():
                if name in comp_dict:
                    curve.setData(ts, comp_dict[name])
        self._apply_comp_mode()

    def _apply_comp_mode(self):
        comp_on = self._comp_toggle.active() == 1
        has_data = self._comp_data is not None
        for name, curve in self._comp_curves.items():
            curve.setVisible(comp_on and has_data and name in self._comp_data)
        self._comp_legend.setVisible(comp_on and has_data)
        if self._comp_key and self._comp_key in self._f_curves:
            btn_on = (self._comp_key in self._legend_btns
                      and self._legend_btns[self._comp_key].isChecked())
            self._f_curves[self._comp_key].setVisible(btn_on and not comp_on)

    def set_noise(self, noise_val: float):
        if np.isfinite(noise_val):
            self._noise_lbl.setText(f"noise_std={noise_val:.4g}")
        else:
            self._noise_lbl.setText("")

    def set_param_text(self, combo_label: str, params: dict | None):
        if params is None:
            self._param_lbl.setText("")
            return
        def _fmt(v: float, is_time: bool) -> str:
            if not np.isfinite(v): return "nan"
            return f"{v:.1f}s" if is_time else (f"{v:.3g}" if abs(v) >= 1e4 else f"{v:.1f}")
        time_params = {"t_slow", "t_fast", "t_bright"}
        parts = [f"{n}={_fmt(params[n], n in time_params)}" for n in PARAM_NAMES if n in params]
        self._param_lbl.setText(f"{combo_label}   " + "   ".join(parts))

    def set_active_only(self, key: str | None):
        """Check only the given key; uncheck all others."""
        for k, btn in self._legend_btns.items():
            btn.setChecked(k == key)

    def clear_traces(self):
        for btn in self._legend_btns.values():
            btn.setChecked(False)

    def select_all_traces(self):
        for btn in self._legend_btns.values():
            btn.setChecked(True)

    def set_curated_bg(self, is_curated: bool):
        if is_curated:
            c = QApplication.instance().palette().color(QPalette.Window)
            bg = (c.red(), c.green(), c.blue())
        else:
            bg = "w"
        self._trace_glw.setBackground(bg)

    def highlight_winner(self, winner_key: str | None):
        """Make the winner button glow with a gold border."""
        for key, btn in self._legend_btns.items():
            if key not in COMBO_KEY_LIST:
                continue
            base_hex = self._colors[key]
            r, g, b  = _pg_color(base_hex)
            luma     = 0.299*r + 0.587*g + 0.114*b
            txt      = "white" if luma < 160 else "black"
            if key == winner_key:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: rgb({r},{g},{b}); color: {txt};
                        border: 2px solid #e6ac00; padding: 1px 7px;
                        font-size: 9pt; font-weight: bold; border-radius: 3px;
                    }}
                    QPushButton:!checked {{ background: #ccc; color: #999; }}
                """)
            else:
                self._apply_btn_style(btn, base_hex)

    def update(self, timestamps, F, baselines, dffs):
        empty = np.array([])
        self._f_curves["F"].setData(timestamps, F)
        new_keys = set(baselines) | set(dffs)
        for key in TRACE_KEYS:
            had = key in self._current_data_keys
            if key in baselines and key in self._f_curves:
                self._f_curves[key].setData(timestamps, baselines[key])
            elif had and key in self._f_curves:
                self._f_curves[key].setData(empty, empty)
            if key in dffs and key in self._dff_curves:
                self._dff_curves[key].setData(timestamps, dffs[key])
            elif had and key in self._dff_curves:
                self._dff_curves[key].setData(empty, empty)
        self._current_data_keys = new_keys


# ── noise criterion bar plot ──────────────────────────────────────────────────

class NoiseCriterionPlot(QWidget):
    """Bar chart: |median(neg residuals)| per combo vs 0.674·σ_noise target."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setFixedHeight(140)
        self._pi  = self._glw.addPlot()
        self._pi.setLabel("left", "a.u.", size="7pt")
        self._pi.setMouseEnabled(x=False, y=True)
        self._pi.setMenuEnabled(False)
        self._pi.showGrid(x=False, y=True, alpha=0.2)

        # X axis: all trace labels (short, long, then combos)
        ax = self._pi.getAxis("bottom")
        ax.setTicks([[(i, TRACE_LABELS[k]) for i, k in enumerate(TRACE_KEYS)]])

        # Persistent target line
        self._target_line = pg.InfiniteLine(
            angle=0, pos=0,
            pen=_make_pen((220, 30, 30), width=2, style=Qt.DashLine),
        )
        self._pi.addItem(self._target_line)

        self._bars: dict[str, pg.BarGraphItem] = {}
        self._text_items: list[pg.TextItem] = []
        self._winner_key: str | None = None
        self._visual_best_key: str | None = None

        layout.addWidget(self._glw)

    def update(self, med_neg: dict, target: float, winner_key: str | None):
        for b in self._bars.values():
            self._pi.removeItem(b)
        self._bars = {}
        for t in self._text_items:
            self._pi.removeItem(t)
        self._text_items = []

        self._winner_key      = winner_key
        self._visual_best_key = None          # reset; set_visual_best() called after

        self._target_line.setValue(target)

        finite_heights = []
        for i, key in enumerate(TRACE_KEYS):
            val = med_neg.get(key, float("nan"))
            if not np.isfinite(val):
                continue
            h = float(val)
            finite_heights.append(h)

            r, g, b = _pg_color(TRACE_COLORS[key])
            is_candidate = key in COMBO_KEYS
            if is_candidate:
                alpha = 230 if key == winner_key else 110
                pen   = pg.mkPen(color=(220, 170, 0), width=3) if key == winner_key \
                        else pg.mkPen(color=(140, 140, 140), width=1)
            else:
                alpha = 60
                pen   = pg.mkPen(color=(140, 140, 140), width=1)

            bar = pg.BarGraphItem(x=[i], height=[h], width=0.6,
                                  brush=pg.mkBrush(r, g, b, alpha), pen=pen)
            self._pi.addItem(bar)
            self._bars[key] = bar

        y_max = max(finite_heights + [target], default=1.0)
        for i, key in enumerate(TRACE_KEYS):
            val = med_neg.get(key, float("nan"))
            if not np.isfinite(val) or target <= 0:
                continue
            h     = float(val)
            ratio = h / target
            star  = "★" if key == winner_key else ""
            ti    = pg.TextItem(
                text=f"{star}{ratio:.2f}",
                anchor=(0.5, 1.0),
                color=(180, 130, 0) if key == winner_key else (80, 80, 80),
            )
            if key == winner_key:
                ti.setFont(pg.QtGui.QFont("Arial", 8, pg.QtGui.QFont.Bold))
            ti.setPos(i, h + y_max * 0.03)
            self._pi.addItem(ti)
            self._text_items.append(ti)

        self._pi.setYRange(0, y_max * 1.20, padding=0)
        self._pi.setXRange(-0.6, len(TRACE_KEYS) - 0.4, padding=0)

    def set_visual_best(self, visual_best_key: str | None):
        self._visual_best_key = visual_best_key
        for key, bar in self._bars.items():
            if key == visual_best_key:
                bar.setOpts(pen=pg.mkPen(color=(220, 30, 30), width=3))
            elif key == self._winner_key:
                bar.setOpts(pen=pg.mkPen(color=(220, 170, 0), width=3))
            else:
                bar.setOpts(pen=pg.mkPen(color=(140, 140, 140), width=1))

    def set_curated_bg(self, is_curated: bool):
        if is_curated:
            c = QApplication.instance().palette().color(QPalette.Window)
            bg = (c.red(), c.green(), c.blue())
        else:
            bg = "w"
        self._glw.setBackground(bg)


# ── jump-to-index label ───────────────────────────────────────────────────────

class _JumpEdit(QLineEdit):
    """Displays as a plain label; click to type a 1-based index, Enter to jump."""
    jumped = pyqtSignal(int)   # emits 0-based index

    _LABEL_STYLE = "QLineEdit { background: transparent; border: none; padding: 0px; }"
    _EDIT_STYLE  = ("QLineEdit { background: white; border: 1px solid #888; "
                    "border-radius: 2px; padding: 0px 3px; }")

    def __init__(self, color: str = "", parent=None):
        super().__init__(parent)
        self._display = ""
        self._color   = color
        self.setFrame(False)
        self.setReadOnly(True)
        self._apply_label_style()
        self.returnPressed.connect(self._on_return)

    def _apply_label_style(self):
        color_rule = f"color: {self._color}; " if self._color else ""
        self.setStyleSheet(
            f"QLineEdit {{ background: transparent; border: none; padding: 0px; {color_rule}}}"
        )

    def set_display(self, text: str):
        self._display = text
        if self.isReadOnly():
            self.setText(text)

    def mousePressEvent(self, ev):
        if self.isReadOnly():
            self.setReadOnly(False)
            self.setStyleSheet(self._EDIT_STYLE)
            self.setText("")
            self.setPlaceholderText(self._display)
        super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self._revert()
        elif ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            ev.accept()            # mark consumed before state changes in _on_return
            super().keyPressEvent(ev)
        else:
            super().keyPressEvent(ev)

    def focusOutEvent(self, ev):
        self._revert()
        super().focusOutEvent(ev)

    def _revert(self):
        self.setReadOnly(True)
        self._apply_label_style()
        self.setText(self._display)

    def _on_return(self):
        try:
            idx = int(self.text().strip()) - 1   # 1-based input → 0-based
            if idx >= 0:
                self.jumped.emit(idx)
        except ValueError:
            pass
        self._revert()


# ── image panel (copied from original) ───────────────────────────────────────

class _BiToggle(QLabel):
    toggled = pyqtSignal()
    def __init__(self, a, b, active=0, parent=None):
        super().__init__(parent)
        self._opts   = (a, b)
        self._active = active % 2
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setStyleSheet(
            "QLabel{border:1px solid #aaa;border-radius:3px;padding:2px 7px;background:#ebebeb;}"
            "QLabel:hover{background:#ddd;border-color:#888;}"
            "QLabel:disabled{background:#f0f0f0;border-color:#ddd;}")
        self._refresh()
    def active(self): return self._active
    def setActive(self, idx, emit=True):
        idx = idx % 2
        if self._active == idx: return
        self._active = idx; self._refresh()
        if emit: self.toggled.emit()
    def toggle(self): self.setActive(1 - self._active)
    def mousePressEvent(self, ev):
        if self.isEnabled():
            self.toggle()
        super().mousePressEvent(ev)
    def changeEvent(self, ev):
        from PyQt5.QtCore import QEvent
        if ev.type() == QEvent.EnabledChange:
            self.setCursor(Qt.ArrowCursor if not self.isEnabled() else Qt.PointingHandCursor)
            self._refresh()
        super().changeEvent(ev)
    def _refresh(self):
        parts = []
        for i, o in enumerate(self._opts):
            if not self.isEnabled():
                parts.append(f"<span style='color:#ccc'>{o}</span>")
            elif i == self._active:
                parts.append(f"<b style='color:#111'>{o}</b>")
            else:
                parts.append(f"<span style='color:#bbb'>{o}</span>")
        self.setText(" / ".join(parts))


class ImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(4)

        self.zoom_toggle = _BiToggle("Zoom", "FOV (B)", active=0)
        self.img_toggle  = _BiToggle("Mean", "Max (N)", active=1)
        self.mask_chk    = QCheckBox("Mask (M)"); self.mask_chk.setChecked(True)
        for w in (self.zoom_toggle, self.img_toggle, self.mask_chk):
            layout.addWidget(w)

        cg = QGroupBox("Contrast"); cgrid = QGridLayout(cg)
        cgrid.setContentsMargins(4,4,4,4); cgrid.setVerticalSpacing(2)
        self._lo = QSlider(Qt.Horizontal); self._hi = QSlider(Qt.Horizontal)
        for sl in (self._lo, self._hi):
            sl.setRange(0, 1000); sl.setFixedHeight(16)
        self._lo.setValue(0); self._hi.setValue(1000)
        self._auto_btn = QPushButton("Auto"); self._auto_btn.setFixedWidth(44)
        cgrid.addWidget(QLabel("Lo"), 0, 0); cgrid.addWidget(self._lo, 0, 1)
        cgrid.addWidget(QLabel("Hi"), 1, 0); cgrid.addWidget(self._hi, 1, 1)
        cgrid.addWidget(self._auto_btn, 0, 2, 2, 1)
        layout.addWidget(cg)

        self.img_plot = pg.PlotWidget()
        self.img_plot.setFixedHeight(220)
        self.img_plot.hideAxis("left"); self.img_plot.hideAxis("bottom")
        self.img_plot.setAspectLocked(True); self.img_plot.setMenuEnabled(False)
        self.img_item = pg.ImageItem(); self.img_plot.addItem(self.img_item)
        layout.addWidget(self.img_plot)

        self._max_img = self._mean_img = self._mask = None
        self.zoom_toggle.toggled.connect(self.auto_contrast)
        self.img_toggle.toggled.connect(self.auto_contrast)
        self._lo.valueChanged.connect(self._redraw)
        self._hi.valueChanged.connect(self._redraw)
        self._auto_btn.clicked.connect(self.auto_contrast)
        self.mask_chk.stateChanged.connect(self._redraw)

    def set_roi(self, max_img, mean_img, mask):
        self._max_img = max_img; self._mean_img = mean_img; self._mask = mask
        self.auto_contrast()

    def toggle_zoom(self): self.zoom_toggle.toggle()
    def toggle_img_mode(self): self.img_toggle.toggle()

    def set_curated_bg(self, is_curated: bool):
        if is_curated:
            c = QApplication.instance().palette().color(QPalette.Window)
            bg = (c.red(), c.green(), c.blue())
        else:
            bg = "w"
        self.img_plot.setBackground(bg)

    def auto_contrast(self):
        if self._max_img is None: return
        fov, mask = self._current_fov_mask()
        px = fov[mask] if mask.any() else fov.ravel()
        lo, hi = float(np.percentile(px, 2)), float(np.percentile(px, 99))
        span = max(hi - lo, 0.05)
        lo = max(lo - 0.05*span, 0.0); hi = min(hi + 0.05*span, 1.0)
        for sl in (self._lo, self._hi): sl.blockSignals(True)
        self._lo.setValue(int(lo*1000)); self._hi.setValue(int(hi*1000))
        for sl in (self._lo, self._hi): sl.blockSignals(False)
        self._redraw()

    def _current_img(self):
        return self._mean_img if self.img_toggle.active() == 0 else self._max_img

    def _current_fov_mask(self):
        fov = self._current_img()
        if self.zoom_toggle.active() == 1:
            return fov, self._mask
        fc, mc, _ = crop_around_mask(fov, self._mask)
        return fc, mc

    def _redraw(self):
        if self._max_img is None: return
        fov, mask = self._current_fov_mask()
        lo = self._lo.value() / 1000.0
        hi = max(self._hi.value() / 1000.0, lo + 1e-3)
        gray8 = (np.clip((fov-lo)/(hi-lo), 0, 1) * 255).astype(np.uint8)
        h, w  = gray8.shape
        rgba  = np.stack([gray8, gray8, gray8, np.full((h,w),255,dtype=np.uint8)], axis=-1)
        if self.mask_chk.isChecked() and mask.any():
            c = _mask_contour(mask)
            rgba[c, 0] = MASK_COLOR[0]; rgba[c, 1] = MASK_COLOR[1]
            rgba[c, 2] = MASK_COLOR[2]; rgba[c, 3] = 255
        self.img_item.setImage(rgba.transpose(1,0,2))
        self.img_plot.autoRange()


# ── metric histograms ─────────────────────────────────────────────────────────

class MetricHistograms(QWidget):
    _ROW_H = 72

    # Hard-coded display bounds for specific metrics; line clamped to these limits.
    _METRIC_BOUNDS: dict = {"drift_metric": (-1.0, 3.0)}

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        self._plots: dict = {}; self._lines: dict = {}; self._all: dict = {}

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setFixedHeight(len(METRIC_NAMES) * self._ROW_H)
        layout.addWidget(self._glw)
        glw = self._glw
        for i, name in enumerate(METRIC_NAMES):
            pi = glw.addPlot(row=i, col=0)
            title = f"log₁₀({name})" if name in METRIC_LOG else name
            pi.setTitle(title, size="8pt"); pi.hideAxis("left")
            pi.getAxis("bottom").setStyle(tickTextOffset=2)
            pi.setMouseEnabled(x=False, y=False); pi.setMenuEnabled(False)
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=1))
            pi.addItem(line)
            self._plots[name] = pi; self._lines[name] = line

    def load_all(self, agg_df):
        self._all = {}
        for name in METRIC_NAMES:
            if name in agg_df.columns:
                v = agg_df[name].to_numpy(float)
                v = v[np.isfinite(v)]
                if name in METRIC_LOG:
                    v = np.log10(v[v > 0])
                if name in self._METRIC_BOUNDS:
                    lo, hi = self._METRIC_BOUNDS[name]
                    v = v[(v >= lo) & (v <= hi)]
                self._all[name] = v
        self._rebuild()

    def _rebuild(self):
        for name, pi in self._plots.items():
            pi.clear(); pi.addItem(self._lines[name])
            v = self._all.get(name)
            if v is None or len(v) == 0: continue
            counts, edges = np.histogram(v, bins=60)
            pi.addItem(pg.BarGraphItem(
                x0=edges[:-1], x1=edges[1:], height=counts,
                brush=pg.mkBrush(136,136,136,180), pen=None))
            if name in self._METRIC_BOUNDS:
                lo, hi = self._METRIC_BOUNDS[name]
                pi.setXRange(lo, hi, padding=0.02)

    def set_curated_bg(self, is_curated: bool):
        if is_curated:
            c = QApplication.instance().palette().color(QPalette.Window)
            bg = (c.red(), c.green(), c.blue())
        else:
            bg = "w"
        self._glw.setBackground(bg)

    def mark_roi(self, row: dict):
        for name in METRIC_NAMES:
            val = float(row.get(name, float("nan")))
            if not np.isfinite(val):
                continue
            if name in METRIC_LOG:
                if val <= 0:
                    continue
                val = np.log10(val)
            if name in self._METRIC_BOUNDS:
                lo, hi = self._METRIC_BOUNDS[name]
                val = max(lo, min(hi, val))
            self._lines[name].setValue(val)


# ── curation panel ────────────────────────────────────────────────────────────

class CurationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(3)

        # row 1: noise winner + call radios | visual best dropdown
        row1 = QHBoxLayout(); row1.setSpacing(6)
        row1.addWidget(QLabel("Noise winner:"))
        self.winner_lbl = QLabel("—")
        self.winner_lbl.setStyleSheet("font-weight:bold; color:#c06000; min-width:40px;")
        row1.addWidget(self.winner_lbl)
        row1.addSpacing(6)
        self._verdict_btns: dict[str, QRadioButton] = {}
        for v in ("good", "maybe", "bad"):
            rb = QRadioButton(v)
            self._verdict_btns[v] = rb
            row1.addWidget(rb)
        row1.addSpacing(20)
        row1.addWidget(QLabel("Visual best:"))
        self.visual_combo = QComboBox()
        self.visual_combo.addItem("—")
        self.visual_combo.addItem("short", "short")
        self.visual_combo.addItem("long", "long")
        for c in COMBOS:
            self.visual_combo.addItem(COMBO_LABEL[c], COMBO_KEY[c])
        self.visual_combo.setFixedWidth(80)
        row1.addWidget(self.visual_combo)
        row1.addStretch()
        root.addLayout(row1)

        # rows 2a/2b: flag checkboxes split into two rows of 4
        self._flag_checks: dict[str, QCheckBox] = {}
        flag_items = list(zip(FLAG_COLS, [lbl for _, lbl in FLAGS]))
        half = len(flag_items) // 2
        for flag_row_items in (flag_items[:half], flag_items[half:]):
            row = QHBoxLayout(); row.setSpacing(6)
            for col, label in flag_row_items:
                cb = QCheckBox(label)
                self._flag_checks[col] = cb
                row.addWidget(cb)
            row.addStretch()
            root.addLayout(row)

        # already-curated indicator (shown above save buttons)
        self._curated_lbl = QLabel("This ROI is already curated")
        self._curated_lbl.setStyleSheet(
            "color: #1a6b1a; font-weight: bold; font-style: italic; padding: 0px 2px;")
        self._curated_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._curated_lbl.setVisible(False)
        root.addWidget(self._curated_lbl)

        # row 3: notes | navigation buttons
        row3 = QHBoxLayout(); row3.setSpacing(4)
        row3.addWidget(QLabel("Notes:"))
        self.notes_edit = QLineEdit()
        row3.addWidget(self.notes_edit, stretch=1)
        row3.addSpacing(8)
        self.prev_btn     = QPushButton("◀ Prev (J)")
        self.next_roi_btn = QPushButton("Next ▶ (K)")
        self.save_btn     = QPushButton("Save (S)")
        self.next_btn     = QPushButton("Save+Next (Enter)")
        for btn in (self.prev_btn, self.next_roi_btn, self.save_btn, self.next_btn):
            btn.setFixedHeight(22)
            row3.addWidget(btn)
        root.addLayout(row3)

    def set_winner(self, winner_key: str | None):
        if winner_key is None:
            self.winner_lbl.setText("—")
        else:
            combo = KEY_COMBO.get(winner_key, ("?", "?"))
            self.winner_lbl.setText(COMBO_LABEL[combo])

    def get_visual_best(self) -> str:
        data = self.visual_combo.currentData()
        if data is not None:
            return data
        text = self.visual_combo.currentText()
        return "—" if text in ("—", "") else text

    def get_verdict(self) -> str:
        for v, rb in self._verdict_btns.items():
            if rb.isChecked():
                return v
        return "—"

    def get_notes(self) -> str:
        return self.notes_edit.text().strip()

    def get_flags(self) -> dict[str, bool]:
        return {col: cb.isChecked() for col, cb in self._flag_checks.items()}

    def set_flags(self, flags: dict):
        for col, cb in self._flag_checks.items():
            cb.setChecked(bool(flags.get(col, False)))

    def set_state(self, visual_best: str, verdict: str, flags: dict | None = None,
                  notes: str = ""):
        idx = self.visual_combo.findData(visual_best)
        if idx < 0:
            idx = self.visual_combo.findText(visual_best)
        self.visual_combo.setCurrentIndex(max(idx, 0))
        for v, rb in self._verdict_btns.items():
            rb.setChecked(v == verdict)
        self.set_flags(flags or {})
        self.notes_edit.setText(notes)

    def clear(self):
        self.visual_combo.setCurrentIndex(0)
        for rb in self._verdict_btns.values():
            rb.setChecked(False)
        for cb in self._flag_checks.values():
            cb.setChecked(False)
        self.notes_edit.clear()

    def set_curated_indicator(self, is_curated: bool):
        self._curated_lbl.setVisible(is_curated)


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(
        self,
        sessions:   list[tuple[str, Path]],
        combo_runs: dict,
        output_path: Path,
        user: str = "",
        agg_df=None,
        roi_list: list | None = None,
    ):
        super().__init__()
        self._sessions    = sessions
        self._combo_runs  = combo_runs
        # Pre-build the hashable tuple needed by load_session's lru_cache
        self._combo_run_strs = tuple(str(combo_runs[c]) for c in COMBOS)
        self._output_path = Path(output_path)
        self._user        = user
        self._sess_idx    = 0
        self._roi_idx     = 0
        self._session_data  = None
        self._curation_df   = load_curation(self._output_path)
        self._current_winner: str | None = None
        self._roi_list = roi_list   # list of (session_key, roi_index) or None
        self._list_pos = 0

        self.setWindowTitle(
            f"binit0 Noise QC — {user}" if user else "binit0 Noise QC"
        )
        self.resize(1250, 760)

        self._build_ui()
        self._wire_signals()

        if agg_df is not None:
            self.metric_hists.load_all(agg_df)

        if self._sessions:
            if self._roi_list:
                self._goto_list_entry(0)
            else:
                self._load_session(0)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4,4,4,4); root.setSpacing(4)

        # top bar
        top = QHBoxLayout()
        top.addWidget(QLabel("Session:"))
        self.sess_combo = QComboBox(); self.sess_combo.setMinimumWidth(280)
        for sk, _ in self._sessions:
            self.sess_combo.addItem(sk)
        top.addWidget(self.sess_combo); top.addSpacing(12)
        self.roi_label   = _JumpEdit()
        self.roi_label.set_display("ROI: — / —")
        self.roi_label.setMinimumWidth(90)
        self.plane_label  = QLabel("")
        self.roiid_label  = QLabel("")
        self.noise_floor_label = QLabel("")
        self.noise_floor_label.setStyleSheet(
            "QLabel { color: #b35c00; font-weight: bold; font-size: 9pt; "
            "background: #fff3cd; border: 1px solid #e6ac00; border-radius: 3px; "
            "padding: 0px 5px; }"
        )
        self.noise_floor_label.setVisible(False)
        self.status_label = QLabel("")
        self.list_label   = _JumpEdit(color="#0055cc")
        self.list_label.setStyleSheet(
            "QLineEdit { background: transparent; border: none; padding: 0px; "
            "color: #0055cc; font-weight: bold; }"
        )
        self.list_label.setMinimumWidth(80)
        for lbl in (self.roi_label, self.plane_label, self.roiid_label,
                    self.noise_floor_label, self.status_label, self.list_label):
            top.addWidget(lbl); top.addSpacing(8)
        top.addStretch()
        self.select_examples_btn = QPushButton("Select Examples")
        self.select_examples_btn.setFixedHeight(22)
        self.load_list_btn = QPushButton("Load List…")
        self.load_list_btn.setFixedHeight(22)
        self.clear_list_btn = QPushButton("Clear List")
        self.clear_list_btn.setFixedHeight(22)
        self.clear_list_btn.setEnabled(False)
        self.capture_btn = QPushButton("Capture (C)")
        self.capture_btn.setFixedHeight(22)
        for btn in (self.select_examples_btn, self.load_list_btn,
                    self.clear_list_btn, self.capture_btn):
            top.addWidget(btn)
        root.addLayout(top)

        # body: horizontal splitter so the right-panel boundary is user-adjustable
        body = QSplitter(Qt.Horizontal)
        self._body_splitter = body

        # left: trace panel on top, noise plot + curation side-by-side on bottom
        left_split = QSplitter(Qt.Vertical)
        self.trace_panel = TracePanel()
        self.trace_panel.init_curves()
        left_split.addWidget(self.trace_panel)

        # bottom row: noise plot (left half) + curation panel (right half)
        bottom_w  = QWidget()
        bottom_lay = QHBoxLayout(bottom_w)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(4)
        self.noise_plot = NoiseCriterionPlot()
        self.curation   = CurationPanel()
        bottom_lay.addWidget(self.noise_plot, stretch=1)
        bottom_lay.addWidget(self.curation,   stretch=1)
        left_split.addWidget(bottom_w)
        left_split.setStretchFactor(0, 1)
        left_split.setStretchFactor(1, 0)
        body.addWidget(left_split)

        # right: image + metric histograms
        right = QWidget(); right.setMinimumWidth(160)
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(4)
        self.image_panel  = ImagePanel(); rl.addWidget(self.image_panel)
        line = QFrame(); line.setFrameShape(QFrame.HLine); rl.addWidget(line)
        self.metric_hists = MetricHistograms(); rl.addWidget(self.metric_hists)
        rl.addStretch()
        body.addWidget(right)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        root.addWidget(body, stretch=1)

        self._capture_dir: Path | None = None

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, '_body_sized', False):
            total = self._body_splitter.width()
            right_w = 240
            self._body_splitter.setSizes([max(total - right_w, 400), right_w])
            self._body_sized = True

    def _wire_signals(self):
        self.sess_combo.currentIndexChanged.connect(self._on_session_changed)
        self.select_examples_btn.clicked.connect(self._open_select_panel)
        self.load_list_btn.clicked.connect(self._load_roi_list)
        self.clear_list_btn.clicked.connect(self._clear_roi_list)
        self.capture_btn.clicked.connect(self._capture)
        self.curation.visual_combo.currentIndexChanged.connect(self._on_visual_best_changed)
        self.curation.prev_btn.clicked.connect(self._prev_roi)
        self.curation.next_roi_btn.clicked.connect(self._next_roi)
        self.curation.save_btn.clicked.connect(self._save)
        self.curation.next_btn.clicked.connect(self._save_and_next)
        self.trace_panel.baseline_mode_changed.connect(self._refresh_roi)
        self.roi_label.jumped.connect(self._jump_to_roi)
        self.list_label.jumped.connect(self._goto_list_entry)

    def _on_visual_best_changed(self):
        vb  = self.curation.get_visual_best()
        key = vb if vb in COMBO_KEYS else None
        # red border only when user chose a combo that differs from the noise winner
        highlight = key if (key is not None and key != self._current_winner) else None
        self.noise_plot.set_visual_best(highlight)

    # ── select-examples panel ─────────────────────────────────────────────────

    def _open_select_panel(self):
        from .select_panel import SelectExamplesWindow
        if not hasattr(self, "_select_win") or self._select_win is None:
            self._select_win = SelectExamplesWindow(
                sessions=self._sessions,
                combo_runs=self._combo_runs,
                parent=self,
            )
            self._select_win.setAttribute(Qt.WA_DeleteOnClose)
            self._select_win.destroyed.connect(
                lambda: setattr(self, "_select_win", None))
        self._select_win.show()
        self._select_win.raise_()
        self._select_win.activateWindow()

    # ── list loading ──────────────────────────────────────────────────────────

    def _load_roi_list(self):
        start = "/scratch"
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI list CSV", start, "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            entries = list(zip(df["session_key"].astype(str),
                               df["roi_index"].astype(int)))
        except Exception as e:
            self.status_label.setText(f"[CSV error: {e}]")
            return
        if not entries:
            self.status_label.setText("[CSV is empty]")
            return
        self._roi_list = entries
        self._list_pos = 0
        self.clear_list_btn.setEnabled(True)
        self.status_label.setText(
            f"Loaded {len(entries)} ROIs from {Path(path).name}")
        self._update_list_label()
        self._goto_list_entry(0)

    def _clear_roi_list(self):
        self._roi_list = None
        self._list_pos = 0
        self.clear_list_btn.setEnabled(False)
        self.list_label.set_display("")
        self.roi_label.setVisible(True)
        self.status_label.setText("List cleared — session mode")
        if self._session_data is not None:
            self._refresh_roi()

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_session(self, idx: int):
        self._sess_idx = idx
        self._roi_idx  = 0
        sess_key, inp_dir = self._sessions[idx]
        self._session_data = load_session(sess_key, str(inp_dir), self._combo_run_strs)
        self._curation_df  = load_curation(self._output_path)
        self.sess_combo.blockSignals(True)
        self.sess_combo.setCurrentIndex(idx)
        self.sess_combo.blockSignals(False)
        self._refresh_roi()

    def _refresh_roi(self):
        sd  = self._session_data
        if sd is None: return
        idx = self._roi_idx
        n   = sd.n_rois
        # In list mode the ROI label is hidden; session/plane/cell info still shown below
        if self._roi_list:
            self.roi_label.setVisible(False)
        else:
            self.roi_label.setVisible(True)
            self.roi_label.set_display(f"ROI: {idx+1}/{n}")

        ts    = sd.timestamps
        F_roi = np.asarray(sd.F[idx])

        # Noise bar (F0trend/IRLS residuals)
        med_neg, target, winner_key = compute_noise_bar(idx, sd, use_f0trend=True)
        self._current_winner = winner_key
        self.noise_plot.update(med_neg, target, winner_key)
        self.trace_panel.highlight_winner(winner_key)
        self.trace_panel.set_noise(float(sd.noise[idx]))
        self.curation.set_winner(winner_key)

        # Build all traces (F0trend/IRLS); only winner shown by default
        baselines: dict = {}
        dffs:      dict = {}
        for key in TRACE_KEYS:
            if key == "short":
                baselines[key] = np.asarray(sd.baselines[key][idx])
                dffs[key]      = np.asarray(sd.dff_short[idx])
            elif key == "long":
                baselines[key] = np.asarray(sd.baselines[key][idx])
                dffs[key]      = np.asarray(sd.dff_long[idx])
            else:
                b = np.asarray(sd.baselines[key][idx])
                baselines[key] = b
                dffs[key]      = _safe_dff(F_roi, b)

        self.trace_panel.set_active_only(winner_key)
        self.trace_panel.update(ts, F_roi, baselines, dffs)

        # Component breakdown + parameter readout for winner
        comps  = (compute_model_components(idx, sd, winner_key)
                  if winner_key and winner_key in sd.res_all else None)
        self.trace_panel.set_component_data(winner_key, sd.timestamps, comps)
        if winner_key and winner_key in sd.res_all:
            raw = sd.res_all[winner_key][idx]
            params = dict(zip(PARAM_NAMES, raw))
            combo_lbl = COMBO_LABEL[KEY_COMBO[winner_key]]
        else:
            params, combo_lbl = None, ""
        self.trace_panel.set_param_text(combo_lbl, params)

        # Noise-floor indicator (per-combo, all combos checked)
        clamped_combos = [
            key for key in COMBO_KEYS
            if key in sd.noise_clamped and idx < len(sd.noise_clamped[key])
            and sd.noise_clamped[key][idx]
        ]
        if clamped_combos:
            labels = ", ".join(clamped_combos)
            self.noise_floor_label.setText(f"noise-floored: {labels}")
            self.noise_floor_label.setVisible(True)
        else:
            self.noise_floor_label.setVisible(False)

        # Image
        row          = sd.rois.iloc[idx]
        plane_id     = str(row["plane_id"])
        cell_roi_id  = int(row["cell_roi_id"])
        self.plane_label.setText(f"plane: {plane_id}")
        self.roiid_label.setText(f"cell_roi_id: {cell_roi_id}")
        try:
            plane = load_plane_assets(str(sd.inputs_dir), plane_id)
            mask  = get_roi_mask(str(sd.inputs_dir), plane_id, cell_roi_id)
            self.image_panel.set_roi(plane.max_norm, plane.mean_norm, mask)
        except Exception as e:
            self.status_label.setText(f"[image error: {e}]")

        # Metrics
        if not sd.metrics.empty:
            self.metric_hists.mark_roi(sd.metrics.iloc[idx].to_dict())

        # Curation state
        dec = lookup_decision(self._curation_df, sd.session_key, idx)
        if dec is not None:
            from .curation import FLAG_COLS as _FLAG_COLS
            flags = {col: (lambda v: False if (isinstance(v, float) and np.isnan(v)) else bool(v))(dec.get(col, False))
                     for col in _FLAG_COLS}
            saved_vb = str(dec.get("visual_best", "—"))
            if saved_vb == winner_key:
                saved_vb = "—"
            self.curation.set_state(
                saved_vb,
                str(dec.get("verdict", "—")),
                flags=flags,
                notes=str(dec.get("notes", "") or ""),
            )
        else:
            self.curation.clear()
            self.curation.set_winner(winner_key)
        self._update_curated_indicator(dec is not None)

    # ── navigation ────────────────────────────────────────────────────────────

    def _prev_roi(self):
        if self._roi_list:
            self._goto_list_entry(self._list_pos - 1)
        elif self._session_data and self._roi_idx > 0:
            self._roi_idx -= 1
            self._refresh_roi()

    def _next_roi(self):
        if self._roi_list:
            self._goto_list_entry(self._list_pos + 1)
        elif self._session_data and self._roi_idx < self._session_data.n_rois - 1:
            self._roi_idx += 1
            self._refresh_roi()

    def _on_session_changed(self, idx): self._load_session(idx)

    def _jump_to_roi(self, idx: int):
        """Jump to a session ROI by 0-based index. Only active in session mode (no list)."""
        if self._roi_list or self._session_data is None:
            return
        self._roi_idx = max(0, min(idx, self._session_data.n_rois - 1))
        self._refresh_roi()

    def _goto_list_entry(self, pos: int):
        if not self._roi_list:
            return
        pos = max(0, min(pos, len(self._roi_list) - 1))
        self._list_pos = pos
        sess_key, roi_idx = self._roi_list[pos]
        for i, (sk, _) in enumerate(self._sessions):
            if sk == sess_key:
                if i != self._sess_idx:
                    self._load_session(i)   # resets _roi_idx=0, calls _refresh_roi
                self._roi_idx = roi_idx
                self._refresh_roi()
                self._update_list_label()
                return
        self.status_label.setText(f"[list: session {sess_key!r} not in runs dir]")

    def _update_list_label(self):
        if self._roi_list:
            self.list_label.set_display(f"List: {self._list_pos + 1}/{len(self._roi_list)}")
        else:
            self.list_label.set_display("")

    # ── curation ─────────────────────────────────────────────────────────────

    def _update_curated_indicator(self, is_curated: bool):
        self.curation.set_curated_indicator(is_curated)
        self.trace_panel.set_curated_bg(is_curated)
        self.noise_plot.set_curated_bg(is_curated)
        self.image_panel.set_curated_bg(is_curated)
        self.metric_hists.set_curated_bg(is_curated)

    def _save(self):
        sd = self._session_data
        if sd is None: return
        idx = self._roi_idx

        # warn before overwriting an existing decision
        if lookup_decision(self._curation_df, sd.session_key, idx) is not None:
            ans = QMessageBox.warning(
                self, "Overwrite curation?",
                "This ROI already has a saved decision.\nReplace it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        row          = sd.rois.iloc[idx]
        noise_winner = (COMBO_LABEL[KEY_COMBO[self._current_winner]]
                        if self._current_winner else "—")
        self._curation_df = save_decision(
            session_key=sd.session_key,
            roi_index=idx,
            plane_id=str(row["plane_id"]),
            cell_roi_id=int(row["cell_roi_id"]),
            noise_winner=noise_winner,
            visual_best=self.curation.get_visual_best(),
            verdict=self.curation.get_verdict(),
            flags=self.curation.get_flags(),
            notes=self.curation.get_notes(),
            user=self._user,
            path=self._output_path,
        )
        self.status_label.setText(f"Saved → {self._output_path}")
        self._update_curated_indicator(True)

    def _save_and_next(self):
        self._save()
        if self._roi_list:
            self._goto_list_entry(self._list_pos + 1)
        else:
            self._next_roi()

    # ── capture ───────────────────────────────────────────────────────────────

    def _resolve_capture_dir(self) -> Path | None:
        if self._capture_dir is not None:
            return self._capture_dir
        default = self._output_path.parent / "captures"
        try:
            default.mkdir(parents=True, exist_ok=True)
            probe = default / ".probe"; probe.touch(); probe.unlink()
            self._capture_dir = default
            return default
        except (PermissionError, OSError):
            pass
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a save directory", str(Path.home()))
        if not chosen: return None
        p = Path(chosen)
        p.mkdir(parents=True, exist_ok=True)
        self._capture_dir = p
        return p

    def _capture(self):
        import datetime
        out = self._resolve_capture_dir()
        if out is None: return
        sd  = self._session_data
        tag = ""
        if sd:
            row = sd.rois.iloc[self._roi_idx]
            tag = f"_{sd.session_key}_{row['plane_id']}_cell{int(row['cell_roi_id'])}"
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out / f"capture{tag}_{ts}.png"
        # grabWindow captures from the screen compositor, so OpenGL widgets
        # render correctly without disrupting the GL framebuffer state.
        screen = QApplication.primaryScreen()
        pix = screen.grabWindow(int(self.winId()))
        pix.save(str(path))
        self.status_label.setText(f"Saved: {path}")

    # ── keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        nav = {Qt.Key_J: self._prev_roi, Qt.Key_K: self._next_roi,
               Qt.Key_S: self._save,     Qt.Key_Return: self._save_and_next,
               Qt.Key_C: self._capture,  Qt.Key_M: self.image_panel.mask_chk.toggle,
               Qt.Key_B: self.image_panel.toggle_zoom,
               Qt.Key_N: self.image_panel.toggle_img_mode,
               Qt.Key_Z: self.trace_panel.clear_traces,
               Qt.Key_A: self.trace_panel.select_all_traces,
               Qt.Key_H: self.trace_panel._home,
               Qt.Key_V: self.trace_panel.toggle_comp_mode}
        if key in nav:
            nav[key]()
            return
        # keys 1-9 toggle traces 1-9; key 0 toggles the last trace (c45)
        qt_num = [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5,
                  Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9]
        for i, qt_k in enumerate(qt_num):
            if key == qt_k and i < len(TRACE_KEYS):
                self.trace_panel.toggle_trace(TRACE_KEYS[i])
                return
        if key == Qt.Key_0:
            self.trace_panel.toggle_trace(TRACE_KEYS[-1])
            return
        super().keyPressEvent(event)


# ── entry point ───────────────────────────────────────────────────────────────

def run(runs_dir: Path | None = None, output: Path | None = None, roi_list: Path | None = None):
    pg.setConfigOptions(background="w", foreground="k", antialias=False, useOpenGL=True)
    app = QApplication.instance() or QApplication(sys.argv)

    # ── pick runs_dir if not supplied ─────────────────────────────────────────
    msg = None
    while runs_dir is None:
        title = ("Select runs directory (contains numbered run sub-folders)"
                 if msg is None else f"⚠ {msg} — pick again, or Cancel to quit")
        _default_start = Path("/root/capsule/data")
        _start = str(_default_start) if _default_start.exists() else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(None, title, _start)
        if not chosen:
            sys.exit(0)
        runs_dir = Path(chosen)
        try:
            combo_runs = discover_combo_runs(runs_dir)
        except Exception as e:
            msg = str(e); runs_dir = None; continue
        missing = [c for c in COMBOS if c not in combo_runs]
        if missing:
            msg = (f"Missing binit0 runs for {missing}. "
                   "Make sure all 8 (c_pos,c_neg) combos are present.")
            runs_dir = None
        else:
            break
    else:
        combo_runs = discover_combo_runs(runs_dir)

    # ── load sessions ─────────────────────────────────────────────────────────
    try:
        sessions = list_sessions(runs_dir, combo_runs)
    except RuntimeError as e:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Setup error", str(e))
        sys.exit(1)

    if not sessions:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(
            None, "No sessions",
            f"No complete sessions found under the input directories.\n"
            f"Runs dir: {runs_dir}\n"
            "Check that metadata.json in each run folder points to a valid inputs_dir.",
        )
        sys.exit(1)

    # ── login ─────────────────────────────────────────────────────────────────
    user = ""
    while not user.strip():
        name, ok = QInputDialog.getText(None, "Login", "Enter your name:")
        if not ok:
            sys.exit(0)
        user = name.strip()

    # ── output path ───────────────────────────────────────────────────────────
    if output is None:
        default = runs_dir / "binit0_qc_curation.csv"
        try:
            default.parent.mkdir(parents=True, exist_ok=True)
            probe = default.parent / ".write_probe"
            probe.touch(); probe.unlink()
            output = default
        except (PermissionError, OSError):
            output = Path("/root/capsule/scratch/binit0_qc_curation.csv")

    print(f"Runs dir  : {runs_dir}")
    print(f"Sessions  : {len(sessions)}")
    print(f"Output    : {output}")
    print("Loading aggregate metrics…")
    agg_df = aggregate_metrics(sessions)

    roi_list_entries = None
    if roi_list is not None:
        import pandas as pd
        rl_df = pd.read_csv(roi_list)
        roi_list_entries = list(zip(
            rl_df["session_key"].astype(str),
            rl_df["roi_index"].astype(int),
        ))
        print(f"ROI list  : {len(roi_list_entries)} entries from {roi_list}")

    win = MainWindow(
        sessions=sessions, combo_runs=combo_runs,
        output_path=output, user=user, agg_df=agg_df,
        roi_list=roi_list_entries,
    )
    win.show()
    sys.exit(app.exec_())
