# -*- coding: utf-8 -*-
"""
Desktop app for Computer Vision Lab — Step 1
Functions implemented:
- Open color image
- Show original
- Show grayscale via RGB averaging
- Show grayscale via HSV (V channel)
UI: minimal PySide6 app with 3 tabs to preview images.
Comments are in English by request.
"""
from __future__ import annotations
import sys
from typing import Optional
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QLabel, QVBoxLayout, QTabWidget, QMessageBox, QSizePolicy, QStatusBar,
    QScrollArea, QToolBar, QComboBox, QHBoxLayout, QPushButton, QDockWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import json

from .image_ops import (
    read_color_bgr, to_gray_mean, to_gray_hsv_value,
    threshold_fixed, threshold_otsu,
    hist_normalize, hist_equalize, hist_contrast_stretch,
    gaussian_blur, laplacian_sharpen, sobel_edges,
    cyclic_shift, rotate_about_center,
)
from .qt_image import cv_to_qpixmap

class ImageLabel(QLabel):
    """
    QLabel that keeps aspect ratio when scaled.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None
        self._fit_to_window = True

    def set_cv_image(self, img: np.ndarray):
        pm = cv_to_qpixmap(img)
        self._pixmap = pm
        if self._fit_to_window:
            self.setPixmap(pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.setPixmap(pm)

    def set_fit_to_window(self, enabled: bool):
        """
        When enabled, the image scales to fit available space while keeping aspect ratio.
        """
        self._fit_to_window = bool(enabled)
        if self._pixmap:
            if self._fit_to_window:
                self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.setPixmap(self._pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap and self._fit_to_window:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Lab — Step 1 (OpenCV + PySide6)")
        self.resize(1200, 800)
        self._bgr_image: Optional[np.ndarray] = None
        self._gray_mean: Optional[np.ndarray] = None
        self._gray_hsv: Optional[np.ndarray] = None
        self._tab_image_map: dict[str, np.ndarray] = {}

        # Central tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.West)  # vertical tabs on the left
        self.setCentralWidget(self.tabs)

        self.label_original = ImageLabel()
        self.label_gray_mean = ImageLabel()
        self.label_gray_hsv = ImageLabel()
        self.label_bin_fixed = ImageLabel()
        self.label_bin_otsu = ImageLabel()
        self.label_hist_norm = ImageLabel()
        self.label_hist_eq = ImageLabel()
        self.label_hist_stretch = ImageLabel()
        self.label_blur = ImageLabel()
        self.label_sharpen = ImageLabel()
        self.label_edges = ImageLabel()
        self.label_shift = ImageLabel()
        self.label_rotate = ImageLabel()

        self.tabs.addTab(self._wrap(self.label_original), "Original (BGR→RGB)")
        self.tabs.addTab(self._wrap(self.label_gray_mean), "Gray — mean(R,G,B)")
        self.tabs.addTab(self._wrap(self.label_gray_hsv), "Gray — HSV (V channel)")
        self.tabs.addTab(self._wrap(self.label_bin_fixed), "Binarize — Fixed thr=128")
        self.tabs.addTab(self._wrap(self.label_bin_otsu), "Binarize — Otsu")
        self.tabs.addTab(self._wrap(self.label_hist_norm), "Normalize")
        self.tabs.addTab(self._wrap(self.label_hist_eq), "Equalize")
        self.tabs.addTab(self._wrap(self.label_hist_stretch), "Contrast stretch")
        self.tabs.addTab(self._wrap(self.label_blur), "Blur — Gaussian (ksize=5)")
        self.tabs.addTab(self._wrap(self.label_sharpen), "Sharpen — Laplacian")
        self.tabs.addTab(self._wrap(self.label_edges), "Edges — Sobel")
        self.tabs.addTab(self._wrap(self.label_shift), "Geometry — Cyclic shift")
        self.tabs.addTab(self._wrap(self.label_rotate), "Geometry — Rotate center +15°")
        # Histogram is shown on demand via the Parameters panel

        # Menu
        self._build_menu()
        self._build_toolbar()
        self._build_tab_navigator()
        self._build_params_dock()

        # Status bar
        self.setStatusBar(QStatusBar())

    def _wrap(self, w: QWidget) -> QWidget:
        # Scrollable container to avoid window growing to image size
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setWidget(w)
        return scroll

    def _build_menu(self):
        open_act = QAction("Open Image…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self.open_image)

        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        export_act = QAction("Export Results…", self)
        export_act.triggered.connect(self._export_results)
        file_menu.addAction(export_act)
        file_menu.addSeparator()
        file_menu.addAction(quit_act)

        # View menu
        view_menu = menubar.addMenu("&View")
        self.act_fit = QAction("Fit to Window", self, checkable=True)
        self.act_fit.setChecked(True)
        self.act_fit.toggled.connect(self._toggle_fit_to_window)
        view_menu.addAction(self.act_fit)

        # No separate Plots menu; histogram is opened from Parameters panel

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        open_act = QAction("Open", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self.open_image)
        tb.addAction(open_act)
        tb.addSeparator()
        self.act_fit_tb = QAction("Fit", self, checkable=True)
        self.act_fit_tb.setChecked(True)
        self.act_fit_tb.toggled.connect(self._toggle_fit_to_window)
        tb.addAction(self.act_fit_tb)
        save_act = QAction("Save", self)
        save_act.setShortcut(QKeySequence.Save)
        save_act.triggered.connect(self._save_current_result)
        tb.addSeparator()
        tb.addAction(save_act)

    def _build_tab_navigator(self):
        # Lightweight navigator under the toolbar
        container = QWidget()
        hl = QHBoxLayout(container)
        hl.setContentsMargins(8, 0, 8, 0)
        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.cmb_tabs = QComboBox()
        self._refresh_tab_names()
        self.btn_prev.clicked.connect(self._go_prev_tab)
        self.btn_next.clicked.connect(self._go_next_tab)
        self.cmb_tabs.currentIndexChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(lambda _: self._sync_tab_selector())
        self.tabs.currentChanged.connect(lambda _: self._update_params_visibility())
        hl.addWidget(self.btn_prev)
        hl.addWidget(self.btn_next)
        hl.addStretch(1)
        hl.addWidget(self.cmb_tabs)
        # Insert below toolbar by adding as another toolbar-like row
        tb2 = QToolBar("Navigator")
        tb2.setMovable(False)
        self.addToolBar(tb2)
        tb2.addWidget(container)

    def _refresh_tab_names(self):
        names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        self.cmb_tabs.clear()
        self.cmb_tabs.addItems(names)

    def _sync_tab_selector(self):
        idx = self.tabs.currentIndex()
        if 0 <= idx < self.cmb_tabs.count():
            self.cmb_tabs.blockSignals(True)
            self.cmb_tabs.setCurrentIndex(idx)
            self.cmb_tabs.blockSignals(False)

    def _go_prev_tab(self):
        idx = self.tabs.currentIndex()
        if idx > 0:
            self.tabs.setCurrentIndex(idx - 1)

    def _go_next_tab(self):
        idx = self.tabs.currentIndex()
        if idx < self.tabs.count() - 1:
            self.tabs.setCurrentIndex(idx + 1)

    def _toggle_fit_to_window(self, enabled: bool):
        # Keep menu and toolbar in sync
        if hasattr(self, 'act_fit') and self.act_fit.isChecked() != enabled:
            self.act_fit.setChecked(enabled)
        if hasattr(self, 'act_fit_tb') and self.act_fit_tb.isChecked() != enabled:
            self.act_fit_tb.setChecked(enabled)
        for lab in (
            self.label_original, self.label_gray_mean, self.label_gray_hsv,
            self.label_bin_fixed, self.label_bin_otsu,
            self.label_hist_norm, self.label_hist_eq, self.label_hist_stretch,
            self.label_blur, self.label_sharpen, self.label_edges,
        ):
            lab.set_fit_to_window(enabled)

    def _build_params_dock(self):
        dock = QDockWidget("Parameters", self)
        dock.setObjectName("ParamsDock")
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(8, 8, 8, 8)

        # Binarization
        self.sb_thr = QSpinBox()
        self.sb_thr.setRange(0, 255)
        self.sb_thr.setValue(128)
        self.cb_inv = QCheckBox("invert")
        box_thr = QWidget()
        self._row_thr_widget = box_thr
        h1 = QHBoxLayout(box_thr)
        h1.setContentsMargins(0, 0, 0, 0)
        h1.addWidget(self.sb_thr)
        h1.addWidget(self.cb_inv)
        self.lbl_thr = QLabel("Threshold:")
        form.addRow(self.lbl_thr, box_thr)

        # Gaussian
        self.sb_gk = QSpinBox()
        self.sb_gk.setRange(3, 99)
        self.sb_gk.setSingleStep(2)
        self.sb_gk.setValue(5)
        self.ds_gs = QDoubleSpinBox()
        self.ds_gs.setRange(0.1, 50.0)
        self.ds_gs.setSingleStep(0.1)
        self.ds_gs.setValue(1.0)
        box_g = QWidget()
        self._row_g_widget = box_g
        h2 = QHBoxLayout(box_g)
        h2.setContentsMargins(0, 0, 0, 0)
        h2.addWidget(self.sb_gk)
        h2.addWidget(self.ds_gs)
        self.lbl_g = QLabel("Gaussian k/sigma:")
        form.addRow(self.lbl_g, box_g)

        # Laplacian sharpen
        self.ds_la = QDoubleSpinBox()
        self.ds_la.setRange(0.0, 5.0)
        self.ds_la.setSingleStep(0.1)
        self.ds_la.setValue(1.0)
        self.sb_lk = QSpinBox()
        self.sb_lk.setRange(1, 31)
        self.sb_lk.setSingleStep(2)
        self.sb_lk.setValue(3)
        box_l = QWidget()
        self._row_l_widget = box_l
        h3 = QHBoxLayout(box_l)
        h3.setContentsMargins(0, 0, 0, 0)
        h3.addWidget(self.ds_la)
        h3.addWidget(self.sb_lk)
        self.lbl_l = QLabel("Laplacian a/ksize:")
        form.addRow(self.lbl_l, box_l)

        # Sobel
        self.sb_sk = QSpinBox()
        self.sb_sk.setRange(1, 31)
        self.sb_sk.setSingleStep(2)
        self.sb_sk.setValue(3)
        self.lbl_s = QLabel("Sobel ksize:")
        self._row_s_widget = self.sb_sk
        form.addRow(self.lbl_s, self._row_s_widget)

        # Shift
        self.sb_dx = QSpinBox()
        self.sb_dx.setRange(-10000, 10000)
        self.sb_dx.setValue(40)
        self.sb_dy = QSpinBox()
        self.sb_dy.setRange(-10000, 10000)
        self.sb_dy.setValue(30)
        box_s = QWidget()
        self._row_shift_widget = box_s
        h4 = QHBoxLayout(box_s)
        h4.setContentsMargins(0, 0, 0, 0)
        h4.addWidget(self.sb_dx)
        h4.addWidget(self.sb_dy)
        self.lbl_shift = QLabel("Shift dx/dy:")
        form.addRow(self.lbl_shift, box_s)

        # Rotate
        self.ds_ang = QDoubleSpinBox()
        self.ds_ang.setRange(-360.0, 360.0)
        self.ds_ang.setSingleStep(1.0)
        self.ds_ang.setValue(15.0)
        self.lbl_ang = QLabel("Rotate angle:")
        self._row_ang_widget = self.ds_ang
        form.addRow(self.lbl_ang, self._row_ang_widget)

        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # Connect signals
        for wid in (
            self.sb_thr, self.cb_inv,
            self.sb_gk, self.ds_gs,
            self.ds_la, self.sb_lk,
            self.sb_sk,
            self.sb_dx, self.sb_dy,
            self.ds_ang,
        ):
            if isinstance(wid, (QSpinBox, QDoubleSpinBox)):
                wid.valueChanged.connect(self.update_views)
            elif isinstance(wid, QCheckBox):
                wid.toggled.connect(self.update_views)

        # Histogram button
        self.btn_hist = QPushButton("Show Histogram")
        self.btn_hist.clicked.connect(self._show_hist_dialog)
        form.addRow("", self.btn_hist)

        # Initial visibility
        self._update_params_visibility()

    # Removed old plot tab methods; histogram shows in a dialog now

    def _set_row_visible(self, label: QLabel, field_widget: QWidget, visible: bool):
        label.setVisible(visible)
        field_widget.setVisible(visible)

    def _update_params_visibility(self):
        # Determine current tab
        name = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() > 0 else ""
        # Default: hide all
        self._set_row_visible(self.lbl_thr, self._row_thr_widget, False)
        self.cb_inv.setVisible(False)
        self._set_row_visible(self.lbl_g, self._row_g_widget, False)
        self._set_row_visible(self.lbl_l, self._row_l_widget, False)
        self._set_row_visible(self.lbl_s, self._row_s_widget, False)
        self._set_row_visible(self.lbl_shift, self._row_shift_widget, False)
        self._set_row_visible(self.lbl_ang, self._row_ang_widget, False)

        # Show context-specific
        if name.startswith("Binarize — Fixed"):
            self._set_row_visible(self.lbl_thr, self._row_thr_widget, True)
            self.cb_inv.setVisible(True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Binarize — Otsu"):
            # Only invert is relevant
            self.cb_inv.setVisible(True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Blur — Gaussian"):
            self._set_row_visible(self.lbl_g, self._row_g_widget, True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Sharpen — Laplacian"):
            self._set_row_visible(self.lbl_l, self._row_l_widget, True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Edges — Sobel"):
            self._set_row_visible(self.lbl_s, self._row_s_widget, True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Geometry — Cyclic shift"):
            self._set_row_visible(self.lbl_shift, self._row_shift_widget, True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Geometry — Rotate"):
            self._set_row_visible(self.lbl_ang, self._row_ang_widget, True)
            self.btn_hist.setVisible(True)
        else:
            self.btn_hist.setVisible(True)

    def _odd(self, k: int) -> int:
        k = int(max(1, k))
        return k if k % 2 == 1 else k + 1

    def _save_current_result(self):
        if self._bgr_image is None:
            return
        idx = self.tabs.currentIndex()
        label = [
            self.label_original, self.label_gray_mean, self.label_gray_hsv,
            self.label_bin_fixed, self.label_bin_otsu,
            self.label_hist_norm, self.label_hist_eq, self.label_hist_stretch,
            self.label_blur, self.label_sharpen, self.label_edges,
            self.label_shift, self.label_rotate,
        ][idx]
        pm = label._pixmap  # trusted internal from ImageLabel
        if pm is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save result", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        pm.save(path)

    def _show_hist_dialog(self):
        if self._bgr_image is None:
            return
        # Determine current image
        tab_name = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() > 0 else ""
        img = self._tab_image_map.get(tab_name)
        if img is None:
            img = self._gray_hsv if self._gray_hsv is not None else None
        if img is None:
            return
        # Convert to grayscale if color
        if img.ndim == 3 and img.shape[2] == 3:
            from .image_ops import to_gray_hsv_value
            gray = to_gray_hsv_value(img)
        else:
            gray = img
        from .image_ops import hist_equalize

        dlg = QDialog(self)
        dlg.setWindowTitle("Histogram")
        fig = Figure(figsize=(5, 3), tight_layout=True)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title("Gray histogram")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.hist(gray.ravel(), bins=256, range=(0, 255), color="#4a90e2", alpha=0.6, label="Current")
        eq = hist_equalize(gray)
        ax.hist(eq.ravel(), bins=256, range=(0, 255), color="#e24a4a", alpha=0.6, label="Equalized")
        ax.legend(loc="upper right")
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(canvas)
        btns = QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        lay.addLayout(btns)
        dlg.setLayout(lay)
        dlg.resize(720, 420)
        dlg.exec()

    def _export_results(self):
        if self._bgr_image is None:
            QMessageBox.information(self, "Export", "Open an image first.")
            return
        dir_path = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not dir_path:
            return
        # Collect images from all tabs
        tabs = [
            ("original", self.label_original),
            ("gray_mean", self.label_gray_mean),
            ("gray_hsv", self.label_gray_hsv),
            ("bin_fixed", self.label_bin_fixed),
            ("bin_otsu", self.label_bin_otsu),
            ("normalize", self.label_hist_norm),
            ("equalize", self.label_hist_eq),
            ("contrast_stretch", self.label_hist_stretch),
            ("blur_gaussian", self.label_blur),
            ("sharpen_laplacian", self.label_sharpen),
            ("edges_sobel", self.label_edges),
            ("shift", self.label_shift),
            ("rotate", self.label_rotate),
        ]
        saved = []
        for name, lab in tabs:
            pm = getattr(lab, "_pixmap", None)
            if pm is None:
                continue
            out = os.path.join(dir_path, f"{name}.png")
            pm.save(out)
            saved.append(out)
        # Save params as JSON
        params = {
            "threshold": int(self.sb_thr.value()) if hasattr(self, 'sb_thr') else 128,
            "invert": bool(self.cb_inv.isChecked()) if hasattr(self, 'cb_inv') else False,
            "gaussian": {"ksize": int(self.sb_gk.value()), "sigma": float(self.ds_gs.value())},
            "laplacian": {"alpha": float(self.ds_la.value()), "ksize": int(self.sb_lk.value())},
            "sobel": {"ksize": int(self.sb_sk.value())},
            "shift": {"dx": int(self.sb_dx.value()), "dy": int(self.sb_dy.value())},
            "rotate": {"angle": float(self.ds_ang.value())},
        }
        try:
            with open(os.path.join(dir_path, "params.json"), "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Export", f"Failed to write params.json: {e}")
        QMessageBox.information(self, "Export", f"Saved {len(saved)} images to:\n{dir_path}")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        try:
            bgr = read_color_bgr(path)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self._bgr_image = bgr
        self._gray_mean = to_gray_mean(bgr)
        self._gray_hsv = to_gray_hsv_value(bgr)
        self.update_views()
        h, w = bgr.shape[:2]
        self.statusBar().showMessage(f"Loaded: {path}  |  Size: {w}×{h}")

    def update_views(self):
        if self._bgr_image is None:
            return
        bgr = self._bgr_image
        gray_mean = self._gray_mean if self._gray_mean is not None else to_gray_mean(bgr)
        gray_hsv = self._gray_hsv if self._gray_hsv is not None else to_gray_hsv_value(bgr)

        # Original
        self.label_original.set_cv_image(bgr)
        self._tab_image_map["Original (BGR→RGB)"] = bgr
        self.label_gray_mean.set_cv_image(gray_mean)
        self._tab_image_map["Gray — mean(R,G,B)"] = gray_mean
        self.label_gray_hsv.set_cv_image(gray_hsv)
        self._tab_image_map["Gray — HSV (V channel)"] = gray_hsv

        # Params
        thr = int(self.sb_thr.value()) if hasattr(self, 'sb_thr') else 128
        inv = bool(self.cb_inv.isChecked()) if hasattr(self, 'cb_inv') else False
        gk = self._odd(int(self.sb_gk.value())) if hasattr(self, 'sb_gk') else 5
        gs = float(self.ds_gs.value()) if hasattr(self, 'ds_gs') else 1.0
        la = float(self.ds_la.value()) if hasattr(self, 'ds_la') else 1.0
        lk = self._odd(int(self.sb_lk.value())) if hasattr(self, 'sb_lk') else 3
        sk = self._odd(int(self.sb_sk.value())) if hasattr(self, 'sb_sk') else 3
        dx = int(self.sb_dx.value()) if hasattr(self, 'sb_dx') else 40
        dy = int(self.sb_dy.value()) if hasattr(self, 'sb_dy') else 30
        ang = float(self.ds_ang.value()) if hasattr(self, 'ds_ang') else 15.0

        # Binarization
        bin_fixed = threshold_fixed(gray_hsv, threshold=thr, invert=inv)
        self.label_bin_fixed.set_cv_image(bin_fixed)
        self._tab_image_map["Binarize — Fixed thr=128"] = bin_fixed
        bin_otsu, _ = threshold_otsu(gray_hsv, invert=inv)
        self.label_bin_otsu.set_cv_image(bin_otsu)
        self._tab_image_map["Binarize — Otsu"] = bin_otsu

        # Histogram
        h_norm = hist_normalize(gray_hsv)
        h_eq = hist_equalize(gray_hsv)
        h_st = hist_contrast_stretch(gray_hsv)
        self.label_hist_norm.set_cv_image(h_norm)
        self._tab_image_map["Normalize"] = h_norm
        self.label_hist_eq.set_cv_image(h_eq)
        self._tab_image_map["Equalize"] = h_eq
        self.label_hist_stretch.set_cv_image(h_st)
        self._tab_image_map["Contrast stretch"] = h_st

        # Convolutions
        blur_img = gaussian_blur(gray_hsv, ksize=gk, sigma=gs)
        self.label_blur.set_cv_image(blur_img)
        self._tab_image_map["Blur — Gaussian (ksize=5)"] = blur_img
        sharp_img = laplacian_sharpen(gray_hsv, alpha=la, ksize=lk)
        self.label_sharpen.set_cv_image(sharp_img)
        self._tab_image_map["Sharpen — Laplacian"] = sharp_img
        edges_img = sobel_edges(gray_hsv, ksize=sk)
        self.label_edges.set_cv_image(edges_img)
        self._tab_image_map["Edges — Sobel"] = edges_img

        # Geometry
        shift_img = cyclic_shift(bgr, dx=dx, dy=dy)
        self.label_shift.set_cv_image(shift_img)
        self._tab_image_map["Geometry — Cyclic shift"] = shift_img
        rotate_img = rotate_about_center(bgr, angle_deg=ang)
        self.label_rotate.set_cv_image(rotate_img)
        self._tab_image_map["Geometry — Rotate center +15°"] = rotate_img

        # No persistent histogram logic; use _show_hist_dialog on demand

def run():
    app = QApplication(sys.argv)
    # Modern look
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()
