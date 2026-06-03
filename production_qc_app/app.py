"""PyQt5 + pyqtgraph QC app for production pipeline dFF output."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QRadioButton, QSizePolicy, QSlider, QSplitter,
    QVBoxLayout, QWidget,
)

from .curation import (
    DEFAULT_PATH, DFF_QUALITY_OPTIONS, QC_LABEL_OPTIONS,
    load_curation, lookup, save_decision,
)
from .data import DATA_DIR, PlaneData, list_planes, list_session_dirs, load_plane
from .rois import crop_around_mask, get_roi_mask, mask_contour, normalize_img

# ── constants ─────────────────────────────────────────────────────────────────

F_PEN        = pg.mkPen(color=(30, 30, 30), width=1)
BASELINE_PEN = pg.mkPen(color=(220, 120, 0), width=2)
DFF_PEN      = pg.mkPen(color=(31, 119, 180), width=1)
EVENTS_PEN   = pg.mkPen(color=(200, 30, 30), width=1)
MASK_COLOR   = (255, 50, 50, 160)   # RGBA for ROI contour

_INVALID_STYLE = (
    "QLabel { background: #c0392b; color: white; font-size: 14pt; "
    "font-weight: bold; padding: 8px 12px; border-radius: 6px; }"
)
_VALID_STYLE = (
    "QLabel { background: #27ae60; color: white; font-size: 11pt; "
    "font-weight: bold; padding: 4px 10px; border-radius: 6px; }"
)
_DFF_BTN_COLORS = {
    "good":                    ("#27ae60", "white"),
    "initial bleaching issue": ("#e67e22", "white"),
    "OK":                      ("#2980b9", "white"),
    "bad":                     ("#c0392b", "white"),
}
_QC_BTN_COLORS = {
    "good":      ("#27ae60", "white"),
    "OK":        ("#2980b9", "white"),
    "ambiguous": ("#8e44ad", "white"),
    "bad":       ("#c0392b", "white"),
}


def _btn_style(bg: str, fg: str, checked: bool) -> str:
    if checked:
        return (
            f"QPushButton {{ background-color: {bg}; color: {fg}; "
            "font-weight: bold; border: 2px solid #222; "
            "border-radius: 4px; padding: 3px 8px; }}"
        )
    return (
        "QPushButton { background-color: #e8e8e8; color: #444; "
        "font-weight: normal; border: 1px solid #aaa; "
        "border-radius: 4px; padding: 3px 8px; } "
        "QPushButton:hover { background-color: #d8d8d8; }"
    )


# ── scroll-Y ViewBox ──────────────────────────────────────────────────────────

class _ShiftYViewBox(pg.ViewBox):
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(ev, axis=1)
        else:
            super().wheelEvent(ev, axis=0)


# ── trace panel ───────────────────────────────────────────────────────────────

class TracePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._glw = pg.GraphicsLayoutWidget()
        self.f_plot = self._glw.addPlot(
            row=0, col=0, viewBox=_ShiftYViewBox(), title="Corrected F  +  Baseline"
        )
        self.dff_plot = self._glw.addPlot(
            row=1, col=0, viewBox=_ShiftYViewBox(), title="dFF  +  Events"
        )
        self.dff_plot.setXLink(self.f_plot)

        for pi in (self.f_plot, self.dff_plot):
            pi.setLabel("bottom", "time (s)")
            pi.showGrid(x=False, y=True, alpha=0.2)
            pi.setDownsampling(auto=True, mode="peak")
            pi.setClipToView(True)
        self.f_plot.setLabel("left", "F (a.u.)")
        self.dff_plot.setLabel("left", "dFF")

        self._glw.ci.layout.setRowStretchFactor(0, 5)
        self._glw.ci.layout.setRowStretchFactor(1, 4)
        layout.addWidget(self._glw)

        # events toggle
        ev_row = QHBoxLayout()
        ev_row.setContentsMargins(4, 0, 4, 2)
        self.events_chk = QCheckBox("Show events")
        self.events_chk.setChecked(True)
        ev_row.addWidget(self.events_chk)
        ev_row.addStretch()
        layout.addLayout(ev_row)

        self._f_curve        = None
        self._baseline_curve = None
        self._dff_curve      = None
        self._events_curve   = None
        self._loaded         = False

        self.events_chk.stateChanged.connect(self._toggle_events)

    def load_roi(
        self,
        ts: np.ndarray,
        F: np.ndarray,
        baseline: np.ndarray,
        dff: np.ndarray,
        events: np.ndarray,
    ) -> None:
        # F + baseline
        self.f_plot.clear()
        self._f_curve        = self.f_plot.plot(ts, F, pen=F_PEN, name="F")
        self._baseline_curve = self.f_plot.plot(ts, baseline, pen=BASELINE_PEN, name="baseline")
        leg_f = self.f_plot.addLegend(offset=(10, 10))
        leg_f.addItem(self._f_curve, "F")
        leg_f.addItem(self._baseline_curve, "baseline")

        # dFF + events
        self.dff_plot.clear()
        self._dff_curve    = self.dff_plot.plot(ts, dff, pen=DFF_PEN, name="dFF")
        self._events_curve = self.dff_plot.plot(ts, events, pen=EVENTS_PEN, name="events")
        self._events_curve.setVisible(self.events_chk.isChecked())
        leg_d = self.dff_plot.addLegend(offset=(10, 10))
        leg_d.addItem(self._dff_curve, "dFF")
        leg_d.addItem(self._events_curve, "events")

    def _toggle_events(self, state):
        if self._events_curve is not None:
            self._events_curve.setVisible(bool(state))

    def home(self):
        self.f_plot.autoRange()
        self.dff_plot.autoRange()


# ── image panel ───────────────────────────────────────────────────────────────

class ImagePanel(QWidget):
    _PAD_STEP = 20
    _PAD_MIN  = 10
    _PAD_MAX  = 250

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(4)

        # controls row
        ctrl = QHBoxLayout()
        self._mean_btn = QPushButton("Mean")
        self._max_btn  = QPushButton("Max")
        self._mean_btn.setCheckable(True)
        self._max_btn.setCheckable(True)
        self._max_btn.setChecked(True)
        self._mean_btn.setFixedHeight(22)
        self._max_btn.setFixedHeight(22)
        self._img_grp = QButtonGroup(self)
        self._img_grp.setExclusive(True)
        self._img_grp.addButton(self._mean_btn)
        self._img_grp.addButton(self._max_btn)
        ctrl.addWidget(QLabel("FOV:"))
        ctrl.addWidget(self._max_btn)
        ctrl.addWidget(self._mean_btn)
        ctrl.addSpacing(8)
        self._zin_btn  = QPushButton("Zoom in  (+)")
        self._zout_btn = QPushButton("Zoom out (−)")
        self._zin_btn.setFixedHeight(22)
        self._zout_btn.setFixedHeight(22)
        ctrl.addWidget(self._zin_btn)
        ctrl.addWidget(self._zout_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # contrast
        cg = QGroupBox("Contrast")
        cgrid = QGridLayout(cg)
        cgrid.setContentsMargins(4, 4, 4, 4)
        cgrid.setVerticalSpacing(2)
        self._lo_sl = QSlider(Qt.Horizontal)
        self._hi_sl = QSlider(Qt.Horizontal)
        for sl in (self._lo_sl, self._hi_sl):
            sl.setRange(0, 1000)
            sl.setFixedHeight(16)
        self._lo_sl.setValue(0)
        self._hi_sl.setValue(1000)
        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setFixedWidth(44)
        cgrid.addWidget(QLabel("Lo"), 0, 0)
        cgrid.addWidget(self._lo_sl, 0, 1)
        cgrid.addWidget(QLabel("Hi"), 1, 0)
        cgrid.addWidget(self._hi_sl, 1, 1)
        cgrid.addWidget(self._auto_btn, 0, 2, 2, 1)
        layout.addWidget(cg)

        # image plot — larger since no histograms
        self.img_plot = pg.PlotWidget()
        self.img_plot.setMinimumSize(380, 380)
        self.img_plot.hideAxis("left")
        self.img_plot.hideAxis("bottom")
        self.img_plot.setAspectLocked(True)
        self.img_plot.setMenuEnabled(False)
        self.img_item = pg.ImageItem()
        self.img_plot.addItem(self.img_item)
        self.img_plot.getViewBox().setMouseEnabled(x=True, y=True)
        layout.addWidget(self.img_plot)

        self._max_img = self._mean_img = self._mask = None
        self._pad = 40

        self._mean_btn.clicked.connect(self._auto_contrast)
        self._max_btn.clicked.connect(self._auto_contrast)
        self._zin_btn.clicked.connect(self._zoom_in)
        self._zout_btn.clicked.connect(self._zoom_out)
        self._lo_sl.valueChanged.connect(self._redraw)
        self._hi_sl.valueChanged.connect(self._redraw)
        self._auto_btn.clicked.connect(self._auto_contrast)

    def set_roi(self, max_img: np.ndarray, mean_img: np.ndarray, mask: np.ndarray) -> None:
        self._max_img = max_img
        self._mean_img = mean_img
        self._mask = mask
        self._pad = 40
        self._auto_contrast()

    def _current_fov(self) -> np.ndarray:
        return self._mean_img if self._mean_btn.isChecked() else self._max_img

    def _cropped(self) -> tuple[np.ndarray, np.ndarray]:
        fov = self._current_fov()
        norm = normalize_img(fov)
        crop_fov, crop_mask, _ = crop_around_mask(norm, self._mask, pad=self._pad)
        return crop_fov, crop_mask

    def _auto_contrast(self):
        if self._max_img is None:
            return
        fov, mask = self._cropped()
        px = fov[mask] if mask.any() else fov.ravel()
        finite = px[np.isfinite(px)]
        if len(finite) == 0:
            return
        lo = float(np.percentile(finite, 2))
        hi = float(np.percentile(finite, 99))
        span = max(hi - lo, 0.05)
        lo = max(lo - 0.05 * span, 0.0)
        hi = min(hi + 0.05 * span, 1.0)
        for sl in (self._lo_sl, self._hi_sl):
            sl.blockSignals(True)
        self._lo_sl.setValue(int(lo * 1000))
        self._hi_sl.setValue(int(hi * 1000))
        for sl in (self._lo_sl, self._hi_sl):
            sl.blockSignals(False)
        self._redraw()

    def _redraw(self):
        if self._max_img is None:
            return
        fov, mask = self._cropped()
        lo = self._lo_sl.value() / 1000.0
        hi = max(self._hi_sl.value() / 1000.0, lo + 1e-3)
        gray8 = (np.clip((fov - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        h, w  = gray8.shape
        rgba  = np.stack(
            [gray8, gray8, gray8, np.full((h, w), 255, dtype=np.uint8)], axis=-1
        )
        if mask.any():
            contour = mask_contour(mask)
            rgba[contour, 0] = MASK_COLOR[0]
            rgba[contour, 1] = MASK_COLOR[1]
            rgba[contour, 2] = MASK_COLOR[2]
            rgba[contour, 3] = 255
        self.img_item.setImage(rgba.transpose(1, 0, 2))
        self.img_plot.autoRange()

    def _zoom_in(self):
        self._pad = max(self._PAD_MIN, self._pad - self._PAD_STEP)
        self._redraw()

    def _zoom_out(self):
        self._pad = min(self._PAD_MAX, self._pad + self._PAD_STEP)
        self._redraw()


# ── exclusive radio-style button group ───────────────────────────────────────

class _RadioButtonBar(QWidget):
    """A row of exclusive toggle buttons with per-option colors."""

    def __init__(self, options: list[str], color_map: dict, parent=None):
        super().__init__(parent)
        self._options   = options
        self._color_map = color_map
        self._buttons: dict[str, QPushButton] = {}
        self._current: str | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for opt in options:
            btn = QPushButton(opt)
            btn.setCheckable(False)
            btn.setFixedHeight(26)
            btn.clicked.connect(lambda checked, o=opt: self._select(o))
            self._buttons[opt] = btn
            row.addWidget(btn)
        row.addStretch()
        self._refresh_all()

    def _select(self, opt: str):
        if self._current == opt:
            self._current = None
        else:
            self._current = opt
        self._refresh_all()

    def _refresh_all(self):
        for opt, btn in self._buttons.items():
            active = (opt == self._current)
            bg, fg = self._color_map.get(opt, ("#e8e8e8", "#444"))
            btn.setStyleSheet(_btn_style(bg, fg, active))

    def get_value(self) -> str:
        return self._current or ""

    def set_value(self, val: str):
        self._current = val if val in self._options else None
        self._refresh_all()

    def clear(self):
        self._current = None
        self._refresh_all()


# ── main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        curation_path: Path = DEFAULT_PATH,
    ):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._curation_path = Path(curation_path)
        self._curation_df   = load_curation(curation_path)

        self._session_dirs = list_session_dirs(data_dir)
        if not self._session_dirs:
            raise RuntimeError(f"No session folders found in {data_dir}")

        self._plane: PlaneData | None = None
        self._roi_idx = 0

        self.setWindowTitle("Production DFF QC")
        self.resize(1400, 860)
        self._build_ui()
        self._wire_signals()
        self._on_session_changed(0)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # ── top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(6)

        top.addWidget(QLabel("Session:"))
        self.sess_combo = QComboBox()
        self.sess_combo.setMinimumWidth(420)
        for d in self._session_dirs:
            self.sess_combo.addItem(d.name)
        top.addWidget(self.sess_combo)

        top.addSpacing(10)
        top.addWidget(QLabel("Plane:"))
        self._plane_btn_row = QHBoxLayout()
        self._plane_btn_row.setSpacing(2)
        self._plane_btns: dict[str, QPushButton] = {}
        self._plane_btn_grp = QButtonGroup(self)
        self._plane_btn_grp.setExclusive(True)
        plane_container = QWidget()
        plane_container.setLayout(self._plane_btn_row)
        top.addWidget(plane_container)

        top.addSpacing(10)
        top.addWidget(QLabel("ROI:"))
        self.roi_label = QLabel("0 / 0")
        self.roi_label.setMinimumWidth(70)
        top.addWidget(self.roi_label)
        self.prev_btn = QPushButton("◀ Prev (J)")
        self.next_btn = QPushButton("Next ▶ (K)")
        self.prev_btn.setFixedHeight(24)
        self.next_btn.setFixedHeight(24)
        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)

        top.addSpacing(10)
        self.progress_lbl = QLabel("Curated: 0 / 0")
        self.progress_lbl.setStyleSheet("color: #555; font-style: italic;")
        top.addWidget(self.progress_lbl)
        top.addStretch()
        root.addLayout(top)

        # ── classification validity banner ────────────────────────────────────
        self.validity_banner = QLabel("─")
        self.validity_banner.setAlignment(Qt.AlignCenter)
        self.validity_banner.setWordWrap(True)
        self.validity_banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.validity_banner.setMinimumHeight(36)
        root.addWidget(self.validity_banner)

        # ── divider ───────────────────────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        root.addWidget(line)

        # ── body splitter ─────────────────────────────────────────────────────
        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)

        # left: trace plots
        self.trace_panel = TracePanel()
        body.addWidget(self.trace_panel)

        # right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 4, 4)
        right_layout.setSpacing(6)

        # image
        self.image_panel = ImagePanel()
        right_layout.addWidget(self.image_panel)

        # ── classification detail ─────────────────────────────────────────────
        self.class_detail_lbl = QLabel("")
        self.class_detail_lbl.setWordWrap(True)
        self.class_detail_lbl.setStyleSheet(
            "QLabel { font-size: 9pt; color: #333; padding: 2px 4px; }"
        )
        right_layout.addWidget(self.class_detail_lbl)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(line2)

        # ── ROI QC checkboxes (below image) ───────────────────────────────────
        qc_grp = QGroupBox("ROI QC")
        qc_row = QHBoxLayout(qc_grp)
        qc_row.setSpacing(6)
        self.qc_bar = _RadioButtonBar(QC_LABEL_OPTIONS, _QC_BTN_COLORS)
        qc_row.addWidget(self.qc_bar)
        right_layout.addWidget(qc_grp)

        # ── DFF quality ───────────────────────────────────────────────────────
        dff_grp = QGroupBox("DFF quality")
        dff_row = QHBoxLayout(dff_grp)
        dff_row.setSpacing(6)
        self.dff_quality_bar = _RadioButtonBar(DFF_QUALITY_OPTIONS, _DFF_BTN_COLORS)
        dff_row.addWidget(self.dff_quality_bar)
        right_layout.addWidget(dff_grp)

        # ── save row ──────────────────────────────────────────────────────────
        save_row = QHBoxLayout()
        self.save_btn      = QPushButton("Save (S)")
        self.save_next_btn = QPushButton("Save + Next (Enter)")
        self.save_btn.setFixedHeight(26)
        self.save_next_btn.setFixedHeight(26)
        self.save_btn.setStyleSheet(
            "QPushButton { background: #2980b9; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 3px 10px; }"
            "QPushButton:hover { background: #1f6899; }"
        )
        self.save_next_btn.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 3px 10px; }"
            "QPushButton:hover { background: #1e8449; }"
        )
        save_row.addWidget(self.save_btn)
        save_row.addWidget(self.save_next_btn)
        save_row.addStretch()
        right_layout.addLayout(save_row)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #1a6b1a; font-style: italic; font-size: 9pt;")
        right_layout.addWidget(self.status_lbl)

        right_layout.addStretch()
        body.addWidget(right)
        body.setSizes([820, 440])

        root.addWidget(body, stretch=1)

    # ── signals ───────────────────────────────────────────────────────────────

    def _wire_signals(self):
        self.sess_combo.currentIndexChanged.connect(self._on_session_changed)
        self.prev_btn.clicked.connect(lambda: self._step_roi(-1))
        self.next_btn.clicked.connect(lambda: self._step_roi(+1))
        self.save_btn.clicked.connect(self._save)
        self.save_next_btn.clicked.connect(self._save_and_next)

    # ── session / plane selection ─────────────────────────────────────────────

    def _on_session_changed(self, idx: int):
        session_dir = self._session_dirs[idx]
        planes = list_planes(session_dir)
        # rebuild plane buttons
        for btn in self._plane_btns.values():
            self._plane_btn_row.removeWidget(btn)
            btn.deleteLater()
        self._plane_btns.clear()
        for b in self._plane_btn_grp.buttons():
            self._plane_btn_grp.removeButton(b)

        for plane_id in planes:
            btn = QPushButton(plane_id)
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            self._plane_btn_grp.addButton(btn)
            self._plane_btns[plane_id] = btn
            self._plane_btn_row.addWidget(btn)
            btn.clicked.connect(
                lambda checked, p=plane_id: self._on_plane_selected(p)
            )
        if planes:
            first_btn = self._plane_btns[planes[0]]
            first_btn.setChecked(True)
            self._on_plane_selected(planes[0])

    def _on_plane_selected(self, plane_id: str):
        session_name = self.sess_combo.currentText()
        session_dir  = self._data_dir / session_name
        self._plane = load_plane(str(session_dir), plane_id)
        self._go_roi(0)

    # ── ROI navigation ────────────────────────────────────────────────────────

    def _step_roi(self, delta: int):
        if self._plane is None:
            return
        self._go_roi(self._roi_idx + delta)

    def _go_roi(self, idx: int):
        if self._plane is None:
            return
        idx = max(0, min(idx, self._plane.n_rois - 1))
        self._roi_idx = idx
        self._refresh()

    def _refresh(self):
        plane = self._plane
        idx   = self._roi_idx
        if plane is None:
            return

        ts       = plane.timestamps
        F        = plane.corrected_F[idx]
        baseline = plane.baseline[idx]
        dff      = plane.dff[idx]
        events   = plane.events[idx]

        # traces
        self.trace_panel.load_roi(ts, F, baseline, dff, events)

        # image
        mask = get_roi_mask(plane, idx)
        self.image_panel.set_roi(plane.max_img, plane.mean_img, mask)

        # validity banner
        is_valid, reasons = plane.is_valid(idx)
        if is_valid:
            self.validity_banner.setStyleSheet(_VALID_STYLE)
            self.validity_banner.setText("✔  VALID  —  soma ✓  |  not dendrite ✓  |  not border ✓")
            self.class_detail_lbl.setText(
                f"soma prob: {plane.soma_pred[idx]}  |  "
                f"dendrite prob: {plane.dendrite_pred[idx]}  |  "
                f"border: {plane.border[idx]}"
            )
        else:
            reason_text = "  •  " + "     •  ".join(reasons)
            self.validity_banner.setStyleSheet(_INVALID_STYLE)
            self.validity_banner.setText(f"✘  INVALID:   {reason_text}")
            self.class_detail_lbl.setText(
                f"soma pred={plane.soma_pred[idx]}  |  "
                f"dendrite pred={plane.dendrite_pred[idx]}  |  "
                f"border={plane.border[idx]}"
            )

        # roi info
        n = plane.n_rois
        self.roi_label.setText(f"{idx}  /  {n - 1}")

        # progress
        session_name = self.sess_combo.currentText()
        n_curated = len(
            self._curation_df[
                (self._curation_df["session"] == session_name)
                & (self._curation_df["plane_id"] == plane.plane_id)
            ]
        )
        self.progress_lbl.setText(f"Curated: {n_curated} / {n}")

        # restore previous decision
        dec = lookup(self._curation_df, session_name, plane.plane_id, idx)
        if dec:
            self.dff_quality_bar.set_value(dec.get("dff_quality", ""))
            self.qc_bar.set_value(dec.get("qc_label", ""))
            self.status_lbl.setText(f"Previously saved: {dec.get('timestamp', '')}")
        else:
            self.dff_quality_bar.clear()
            self.qc_bar.clear()
            self.status_lbl.setText("")

    # ── save ─────────────────────────────────────────────────────────────────

    def _save(self):
        if self._plane is None:
            return
        session_name = self.sess_combo.currentText()
        self._curation_df = save_decision(
            session     = session_name,
            plane_id    = self._plane.plane_id,
            roi_index   = self._roi_idx,
            dff_quality = self.dff_quality_bar.get_value(),
            qc_label    = self.qc_bar.get_value(),
            path        = self._curation_path,
        )
        n_curated = len(
            self._curation_df[
                (self._curation_df["session"] == session_name)
                & (self._curation_df["plane_id"] == self._plane.plane_id)
            ]
        )
        self.progress_lbl.setText(f"Curated: {n_curated} / {self._plane.n_rois}")
        dq = self.dff_quality_bar.get_value() or "—"
        ql = self.qc_bar.get_value() or "—"
        self.status_lbl.setText(
            f"✔ Saved ROI {self._roi_idx}  [dff: {dq}  |  qc: {ql}]"
        )

    def _save_and_next(self):
        self._save()
        self._step_roi(+1)

    # ── keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, ev):
        key = ev.key()
        if key in (Qt.Key_J,):
            self._step_roi(-1)
        elif key in (Qt.Key_K,):
            self._step_roi(+1)
        elif key == Qt.Key_S:
            self._save()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._save_and_next()
        else:
            super().keyPressEvent(ev)
