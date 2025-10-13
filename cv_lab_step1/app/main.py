# -*- coding: utf-8 -*-
"""
Desktop app for Computer Vision Lab — Steps 1 & 2
Functions implemented:
Lab 1:
- Open color image, grayscale conversion (RGB mean, HSV)
- Binarization (fixed threshold, Otsu)
- Histogram operations (normalize, equalize, stretch)
- Convolutions (Gaussian blur, Laplacian sharpen, Sobel edges)
- Geometric transforms (cyclic shift, rotation)

Lab 2:
- Hough line detection
- Hough circle detection  
- Local statistical features (mean, variance, skewness, kurtosis)
- Texture-based segmentation with seed point
UI: PySide6 app with tabs and interactive parameters.
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
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, QSlider
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
    hough_lines, hough_circles, local_statistics, texture_segmentation,
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
        self._apply_pixmap()

    def set_fit_to_window(self, enabled: bool):
        """
        When enabled, the image scales to fit available space while keeping aspect ratio.
        """
        self._fit_to_window = bool(enabled)
        self._apply_pixmap()

    def _apply_pixmap(self):
        if not self._pixmap:
            return
        if self._fit_to_window:
            # If inside a QScrollArea, scale to its viewport size
            parent = self.parent()
            viewport_size = None
            if parent and hasattr(parent, 'parent'):
                p2 = parent.parent()
                if p2 and hasattr(p2, 'viewport'):
                    vp = p2.viewport()
                    viewport_size = vp.size()
            target_size = viewport_size if viewport_size is not None else self.size()
            self.setPixmap(self._pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.setPixmap(self._pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap and self._fit_to_window:
            self._apply_pixmap()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Lab — Steps 1 & 2 (OpenCV + PySide6)")
        self.resize(1200, 800)
        self._bgr_image: Optional[np.ndarray] = None
        self._gray_mean: Optional[np.ndarray] = None
        self._gray_hsv: Optional[np.ndarray] = None
        self._tab_image_map: dict[str, np.ndarray] = {}

        # Central tabs (top, more standard UI)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
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

        # Lab 2: Feature Detection
        self.label_hough_lines = ImageLabel()
        self.label_hough_circles = ImageLabel()
        self.label_local_mean = ImageLabel()
        self.label_local_var = ImageLabel()
        self.label_local_skew = ImageLabel()
        self.label_local_kurt = ImageLabel()
        self.label_texture_seg = ImageLabel()

        self.tabs.addTab(self._wrap(self.label_original, "Original (BGR→RGB)"), "Original (BGR→RGB)")
        self.tabs.addTab(self._wrap(self.label_gray_mean, "Gray — mean(R,G,B)"), "Gray — mean(R,G,B)")
        self.tabs.addTab(self._wrap(self.label_gray_hsv, "Gray — HSV (V channel)"), "Gray — HSV (V channel)")
        self.tabs.addTab(self._wrap(self.label_bin_fixed, "Binarize — Fixed thr=128"), "Binarize — Fixed thr=128")
        self.tabs.addTab(self._wrap(self.label_bin_otsu, "Binarize — Otsu"), "Binarize — Otsu")
        self.tabs.addTab(self._wrap(self.label_hist_norm, "Normalize"), "Normalize")
        self.tabs.addTab(self._wrap(self.label_hist_eq, "Equalize"), "Equalize")
        self.tabs.addTab(self._wrap(self.label_hist_stretch, "Contrast stretch"), "Contrast stretch")
        self.tabs.addTab(self._wrap(self.label_blur, "Blur — Gaussian (ksize=5)"), "Blur — Gaussian (ksize=5)")
        self.tabs.addTab(self._wrap(self.label_sharpen, "Sharpen — Laplacian"), "Sharpen — Laplacian")
        self.tabs.addTab(self._wrap(self.label_edges, "Edges — Sobel"), "Edges — Sobel")
        self.tabs.addTab(self._wrap(self.label_shift, "Geometry — Cyclic shift"), "Geometry — Cyclic shift")
        self.tabs.addTab(self._wrap(self.label_rotate, "Geometry — Rotate center +15°"), "Geometry — Rotate center +15°")
        
        # Lab 2 tabs
        self.tabs.addTab(self._wrap(self.label_hough_lines, "Hough Lines"), "Hough Lines")
        self.tabs.addTab(self._wrap(self.label_hough_circles, "Hough Circles"), "Hough Circles")
        self.tabs.addTab(self._wrap(self.label_local_mean, "Local Statistics — Mean"), "Local Statistics — Mean")
        self.tabs.addTab(self._wrap(self.label_local_var, "Local Statistics — Variance"), "Local Statistics — Variance")
        self.tabs.addTab(self._wrap(self.label_local_skew, "Local Statistics — Skewness"), "Local Statistics — Skewness")
        self.tabs.addTab(self._wrap(self.label_local_kurt, "Local Statistics — Kurtosis"), "Local Statistics — Kurtosis")
        self.tabs.addTab(self._wrap(self.label_texture_seg, "Texture Segmentation"), "Texture Segmentation")
        # Histogram is shown on demand via the Parameters panel

        # Menu
        self._build_menu()
        self._build_toolbar()
        self._build_tab_navigator()
        self._build_params_dock()

        # Status bar
        self.setStatusBar(QStatusBar())

    def _wrap(self, img_label: QLabel, title: str) -> QWidget:
        # Scrollable container; put title + image in a container that resizes with viewport
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(12, 12, 12, 12)
        # Title
        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_lbl.setStyleSheet(
            "font-size: 15px; font-weight: 600; padding: 4px 8px;"
            "border-left: 4px solid #2d7eea;"
        )
        lay.addWidget(title_lbl)
        # Image area
        img_wrap = QWidget()
        img_wrap.setStyleSheet("background-color: #111; border-radius: 6px;")
        img_lay = QVBoxLayout(img_wrap)
        img_lay.setContentsMargins(6, 6, 6, 6)
        img_lay.addWidget(img_label)
        lay.addWidget(img_wrap)
        scroll.setWidget(container)
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
        # View menu removed; Fit toggle lives in toolbar only

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
        self.act_fit_tb = QAction("Fit to Window", self, checkable=True)
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
        # Keep toolbar state in sync and update label
        if hasattr(self, 'act_fit_tb') and self.act_fit_tb.isChecked() != enabled:
            self.act_fit_tb.setChecked(enabled)
        # Update toolbar action text to reflect the inverse action (for clarity)
        if hasattr(self, 'act_fit_tb'):
            self.act_fit_tb.setText("Actual Size" if enabled else "Fit to Window")
        for lab in (
            self.label_original, self.label_gray_mean, self.label_gray_hsv,
            self.label_bin_fixed, self.label_bin_otsu,
            self.label_hist_norm, self.label_hist_eq, self.label_hist_stretch,
            self.label_blur, self.label_sharpen, self.label_edges,
            self.label_shift, self.label_rotate,
            self.label_hough_lines, self.label_hough_circles,
            self.label_local_mean, self.label_local_var, self.label_local_skew, self.label_local_kurt,
            self.label_texture_seg,
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
        self.sl_thr = QSlider(Qt.Horizontal)
        self.sl_thr.setRange(0, 255)
        self.sl_thr.setValue(128)
        self.sl_thr.setFixedWidth(160)
        h1.addWidget(self.sb_thr)
        h1.addWidget(self.cb_inv)
        h1.addWidget(self.sl_thr)
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
        self.sl_gk = QSlider(Qt.Horizontal)
        self.sl_gk.setRange(3, 99)
        self.sl_gk.setSingleStep(2)
        self.sl_gk.setPageStep(2)
        self.sl_gk.setValue(5)
        self.sl_gk.setFixedWidth(140)
        self.sl_gs = QSlider(Qt.Horizontal)
        self.sl_gs.setRange(1, 500)  # maps to 0.1..50.0
        self.sl_gs.setValue(10)      # 1.0
        self.sl_gs.setFixedWidth(140)
        h2.addWidget(self.sb_gk)
        h2.addWidget(self.ds_gs)
        h2.addWidget(self.sl_gk)
        h2.addWidget(self.sl_gs)
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
        self.sl_la = QSlider(Qt.Horizontal)
        self.sl_la.setRange(0, 50)  # maps to 0.0..5.0
        self.sl_la.setValue(10)     # 1.0
        self.sl_la.setFixedWidth(140)
        self.sl_lk = QSlider(Qt.Horizontal)
        self.sl_lk.setRange(1, 31)
        self.sl_lk.setSingleStep(2)
        self.sl_lk.setPageStep(2)
        self.sl_lk.setValue(3)
        self.sl_lk.setFixedWidth(140)
        h3.addWidget(self.ds_la)
        h3.addWidget(self.sb_lk)
        h3.addWidget(self.sl_la)
        h3.addWidget(self.sl_lk)
        self.lbl_l = QLabel("Laplacian a/ksize:")
        form.addRow(self.lbl_l, box_l)

        # Sobel
        self.sb_sk = QSpinBox()
        self.sb_sk.setRange(1, 31)
        self.sb_sk.setSingleStep(2)
        self.sb_sk.setValue(3)
        self.lbl_s = QLabel("Sobel ksize:")
        sbox = QWidget(); self._row_s_widget = sbox
        hs = QHBoxLayout(sbox); hs.setContentsMargins(0, 0, 0, 0)
        self.sl_sk = QSlider(Qt.Horizontal)
        self.sl_sk.setRange(1, 31)
        self.sl_sk.setSingleStep(2)
        self.sl_sk.setPageStep(2)
        self.sl_sk.setValue(3)
        self.sl_sk.setFixedWidth(200)
        hs.addWidget(self.sb_sk)
        hs.addWidget(self.sl_sk)
        form.addRow(self.lbl_s, sbox)

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
        self.sl_dx = QSlider(Qt.Horizontal); self.sl_dx.setRange(-1000, 1000); self.sl_dx.setValue(40); self.sl_dx.setFixedWidth(180)
        self.sl_dy = QSlider(Qt.Horizontal); self.sl_dy.setRange(-1000, 1000); self.sl_dy.setValue(30); self.sl_dy.setFixedWidth(180)
        h4.addWidget(self.sb_dx)
        h4.addWidget(self.sb_dy)
        h4.addWidget(self.sl_dx)
        h4.addWidget(self.sl_dy)
        self.lbl_shift = QLabel("Shift dx/dy:")
        form.addRow(self.lbl_shift, box_s)

        # Rotate
        self.ds_ang = QDoubleSpinBox()
        self.ds_ang.setRange(-360.0, 360.0)
        self.ds_ang.setSingleStep(1.0)
        self.ds_ang.setValue(15.0)
        self.lbl_ang = QLabel("Rotate angle:")
        ang_box = QWidget(); self._row_ang_widget = ang_box
        ha = QHBoxLayout(ang_box); ha.setContentsMargins(0, 0, 0, 0)
        self.sl_ang = QSlider(Qt.Horizontal); self.sl_ang.setRange(-180, 180); self.sl_ang.setValue(15); self.sl_ang.setFixedWidth(220)
        ha.addWidget(self.ds_ang)
        ha.addWidget(self.sl_ang)
        form.addRow(self.lbl_ang, ang_box)

        # Lab 2: Hough Lines parameters
        self.cb_hough_adaptive = QCheckBox("Adaptive")
        self.cb_hough_adaptive.setChecked(True)
        
        self.sb_hough_thr = QSpinBox()
        self.sb_hough_thr.setRange(10, 500)
        self.sb_hough_thr.setValue(100)
        self.sl_hough_thr = QSlider(Qt.Horizontal)
        self.sl_hough_thr.setRange(10, 500)
        self.sl_hough_thr.setValue(100)
        self.sl_hough_thr.setFixedWidth(120)
        
        self.sb_hough_min_len = QSpinBox()
        self.sb_hough_min_len.setRange(10, 200)
        self.sb_hough_min_len.setValue(50)
        self.sl_hough_min_len = QSlider(Qt.Horizontal)
        self.sl_hough_min_len.setRange(10, 200)
        self.sl_hough_min_len.setValue(50)
        self.sl_hough_min_len.setFixedWidth(120)
        
        self.sb_hough_max_gap = QSpinBox()
        self.sb_hough_max_gap.setRange(5, 50)
        self.sb_hough_max_gap.setValue(10)
        self.sl_hough_max_gap = QSlider(Qt.Horizontal)
        self.sl_hough_max_gap.setRange(5, 50)
        self.sl_hough_max_gap.setValue(10)
        self.sl_hough_max_gap.setFixedWidth(120)
        
        # Hough Lines - вертикальное расположение
        self.lbl_hough = QLabel("Hough Lines:")
        form.addRow(self.lbl_hough, self.cb_hough_adaptive)
        
        # Threshold
        thr_box = QWidget()
        hh1 = QHBoxLayout(thr_box)
        hh1.setContentsMargins(0, 0, 0, 0)
        hh1.addWidget(QLabel("Threshold:"))
        hh1.addWidget(self.sb_hough_thr)
        hh1.addWidget(self.sl_hough_thr)
        form.addRow("", thr_box)
        
        # Min length
        min_box = QWidget()
        hh2 = QHBoxLayout(min_box)
        hh2.setContentsMargins(0, 0, 0, 0)
        hh2.addWidget(QLabel("Min length:"))
        hh2.addWidget(self.sb_hough_min_len)
        hh2.addWidget(self.sl_hough_min_len)
        form.addRow("", min_box)
        
        # Max gap
        gap_box = QWidget()
        hh3 = QHBoxLayout(gap_box)
        hh3.setContentsMargins(0, 0, 0, 0)
        hh3.addWidget(QLabel("Max gap:"))
        hh3.addWidget(self.sb_hough_max_gap)
        hh3.addWidget(self.sl_hough_max_gap)
        form.addRow("", gap_box)
        
        self._row_hough_widget = [thr_box, min_box, gap_box]

        # Lab 2: Hough Circles parameters
        self.cb_circle_adaptive = QCheckBox("Adaptive")
        self.cb_circle_adaptive.setChecked(True)
        
        self.ds_circle_dp = QDoubleSpinBox()
        self.ds_circle_dp.setRange(0.5, 3.0)
        self.ds_circle_dp.setSingleStep(0.1)
        self.ds_circle_dp.setValue(1.0)
        self.sl_circle_dp = QSlider(Qt.Horizontal)
        self.sl_circle_dp.setRange(5, 30)  # 0.5 to 3.0
        self.sl_circle_dp.setValue(10)     # 1.0
        self.sl_circle_dp.setFixedWidth(100)
        
        self.sb_circle_min_dist = QSpinBox()
        self.sb_circle_min_dist.setRange(10, 200)
        self.sb_circle_min_dist.setValue(50)
        self.sl_circle_min_dist = QSlider(Qt.Horizontal)
        self.sl_circle_min_dist.setRange(10, 200)
        self.sl_circle_min_dist.setValue(50)
        self.sl_circle_min_dist.setFixedWidth(100)
        
        self.sb_circle_param1 = QSpinBox()
        self.sb_circle_param1.setRange(10, 200)
        self.sb_circle_param1.setValue(50)
        self.sl_circle_param1 = QSlider(Qt.Horizontal)
        self.sl_circle_param1.setRange(10, 200)
        self.sl_circle_param1.setValue(50)
        self.sl_circle_param1.setFixedWidth(100)
        
        self.sb_circle_param2 = QSpinBox()
        self.sb_circle_param2.setRange(10, 200)
        self.sb_circle_param2.setValue(30)
        self.sl_circle_param2 = QSlider(Qt.Horizontal)
        self.sl_circle_param2.setRange(10, 200)
        self.sl_circle_param2.setValue(30)
        self.sl_circle_param2.setFixedWidth(100)
        
        # Hough Circles - вертикальное расположение
        self.lbl_circle = QLabel("Hough Circles:")
        form.addRow(self.lbl_circle, self.cb_circle_adaptive)
        
        # DP
        dp_box = QWidget()
        hc1 = QHBoxLayout(dp_box)
        hc1.setContentsMargins(0, 0, 0, 0)
        hc1.addWidget(QLabel("DP:"))
        hc1.addWidget(self.ds_circle_dp)
        hc1.addWidget(self.sl_circle_dp)
        form.addRow("", dp_box)
        
        # Min distance
        dist_box = QWidget()
        hc2 = QHBoxLayout(dist_box)
        hc2.setContentsMargins(0, 0, 0, 0)
        hc2.addWidget(QLabel("Min dist:"))
        hc2.addWidget(self.sb_circle_min_dist)
        hc2.addWidget(self.sl_circle_min_dist)
        form.addRow("", dist_box)
        
        # Param1
        p1_box = QWidget()
        hc3 = QHBoxLayout(p1_box)
        hc3.setContentsMargins(0, 0, 0, 0)
        hc3.addWidget(QLabel("Param1:"))
        hc3.addWidget(self.sb_circle_param1)
        hc3.addWidget(self.sl_circle_param1)
        form.addRow("", p1_box)
        
        # Param2
        p2_box = QWidget()
        hc4 = QHBoxLayout(p2_box)
        hc4.setContentsMargins(0, 0, 0, 0)
        hc4.addWidget(QLabel("Param2:"))
        hc4.addWidget(self.sb_circle_param2)
        hc4.addWidget(self.sl_circle_param2)
        form.addRow("", p2_box)
        
        self._row_circle_widget = [dp_box, dist_box, p1_box, p2_box]

        # Lab 2: Local Statistics parameters
        self.sb_local_window = QSpinBox()
        self.sb_local_window.setRange(5, 51)
        self.sb_local_window.setSingleStep(2)
        self.sb_local_window.setValue(15)
        self.sl_local_window = QSlider(Qt.Horizontal)
        self.sl_local_window.setRange(5, 51)
        self.sl_local_window.setSingleStep(2)
        self.sl_local_window.setValue(15)
        self.sl_local_window.setFixedWidth(150)
        
        local_box = QWidget()
        self._row_local_widget = local_box
        hl = QHBoxLayout(local_box)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.sb_local_window)
        hl.addWidget(self.sl_local_window)
        self.lbl_local = QLabel("Local Stats window:")
        form.addRow(self.lbl_local, self._row_local_widget)

        # Lab 2: Texture Segmentation parameters
        self.sb_texture_thr = QSpinBox()
        self.sb_texture_thr.setRange(5, 100)
        self.sb_texture_thr.setValue(30)
        self.sl_texture_thr = QSlider(Qt.Horizontal)
        self.sl_texture_thr.setRange(5, 100)
        self.sl_texture_thr.setValue(30)
        self.sl_texture_thr.setFixedWidth(100)
        
        self.sb_texture_window = QSpinBox()
        self.sb_texture_window.setRange(5, 51)
        self.sb_texture_window.setSingleStep(2)
        self.sb_texture_window.setValue(15)
        self.sl_texture_window = QSlider(Qt.Horizontal)
        self.sl_texture_window.setRange(5, 51)
        self.sl_texture_window.setSingleStep(2)
        self.sl_texture_window.setValue(15)
        self.sl_texture_window.setFixedWidth(100)
        
        self.sb_seed_x = QSpinBox()
        self.sb_seed_x.setRange(0, 2000)
        self.sb_seed_x.setValue(100)
        self.sl_seed_x = QSlider(Qt.Horizontal)
        self.sl_seed_x.setRange(0, 2000)
        self.sl_seed_x.setValue(100)
        self.sl_seed_x.setFixedWidth(100)
        
        self.sb_seed_y = QSpinBox()
        self.sb_seed_y.setRange(0, 2000)
        self.sb_seed_y.setValue(100)
        self.sl_seed_y = QSlider(Qt.Horizontal)
        self.sl_seed_y.setRange(0, 2000)
        self.sl_seed_y.setValue(100)
        self.sl_seed_y.setFixedWidth(100)
        
        # Texture Segmentation - вертикальное расположение
        self.lbl_texture = QLabel("Texture Segmentation:")
        form.addRow(self.lbl_texture, QLabel(""))
        
        # Threshold
        thr_box = QWidget()
        ht1 = QHBoxLayout(thr_box)
        ht1.setContentsMargins(0, 0, 0, 0)
        ht1.addWidget(QLabel("Threshold:"))
        ht1.addWidget(self.sb_texture_thr)
        ht1.addWidget(self.sl_texture_thr)
        form.addRow("", thr_box)
        
        # Window
        win_box = QWidget()
        ht2 = QHBoxLayout(win_box)
        ht2.setContentsMargins(0, 0, 0, 0)
        ht2.addWidget(QLabel("Window:"))
        ht2.addWidget(self.sb_texture_window)
        ht2.addWidget(self.sl_texture_window)
        form.addRow("", win_box)
        
        # Seed X
        x_box = QWidget()
        ht3 = QHBoxLayout(x_box)
        ht3.setContentsMargins(0, 0, 0, 0)
        ht3.addWidget(QLabel("Seed X:"))
        ht3.addWidget(self.sb_seed_x)
        ht3.addWidget(self.sl_seed_x)
        form.addRow("", x_box)
        
        # Seed Y
        y_box = QWidget()
        ht4 = QHBoxLayout(y_box)
        ht4.setContentsMargins(0, 0, 0, 0)
        ht4.addWidget(QLabel("Seed Y:"))
        ht4.addWidget(self.sb_seed_y)
        ht4.addWidget(self.sl_seed_y)
        form.addRow("", y_box)
        
        self._row_texture_widget = [thr_box, win_box, x_box, y_box]

        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # Connect signals (spinboxes/checkbox)
        self.sb_thr.valueChanged.connect(self.update_views)
        self.cb_inv.toggled.connect(self.update_views)
        
        # Connect adaptive checkboxes to update parameters
        self.cb_hough_adaptive.toggled.connect(self._update_hough_parameters)
        self.cb_circle_adaptive.toggled.connect(self._update_circle_parameters)
        self.sb_gk.valueChanged.connect(self.update_views)
        self.ds_gs.valueChanged.connect(self.update_views)
        self.ds_la.valueChanged.connect(self.update_views)
        self.sb_lk.valueChanged.connect(self.update_views)
        self.sb_sk.valueChanged.connect(self.update_views)
        self.sb_dx.valueChanged.connect(self.update_views)
        self.sb_dy.valueChanged.connect(self.update_views)
        self.ds_ang.valueChanged.connect(self.update_views)
        
        # Lab 2 signals
        self.cb_hough_adaptive.toggled.connect(self.update_views)
        self.sb_hough_thr.valueChanged.connect(self.update_views)
        self.sl_hough_thr.valueChanged.connect(self.sb_hough_thr.setValue)
        self.sb_hough_thr.valueChanged.connect(self.sl_hough_thr.setValue)
        
        self.sb_hough_min_len.valueChanged.connect(self.update_views)
        self.sl_hough_min_len.valueChanged.connect(self.sb_hough_min_len.setValue)
        self.sb_hough_min_len.valueChanged.connect(self.sl_hough_min_len.setValue)
        
        self.sb_hough_max_gap.valueChanged.connect(self.update_views)
        self.sl_hough_max_gap.valueChanged.connect(self.sb_hough_max_gap.setValue)
        self.sb_hough_max_gap.valueChanged.connect(self.sl_hough_max_gap.setValue)
        
        self.cb_circle_adaptive.toggled.connect(self.update_views)
        self.ds_circle_dp.valueChanged.connect(self.update_views)
        self.sl_circle_dp.valueChanged.connect(lambda v: self.ds_circle_dp.setValue(v / 10.0))
        self.ds_circle_dp.valueChanged.connect(lambda v: self.sl_circle_dp.setValue(int(v * 10)))
        
        self.sb_circle_min_dist.valueChanged.connect(self.update_views)
        self.sl_circle_min_dist.valueChanged.connect(self.sb_circle_min_dist.setValue)
        self.sb_circle_min_dist.valueChanged.connect(self.sl_circle_min_dist.setValue)
        
        self.sb_circle_param1.valueChanged.connect(self.update_views)
        self.sl_circle_param1.valueChanged.connect(self.sb_circle_param1.setValue)
        self.sb_circle_param1.valueChanged.connect(self.sl_circle_param1.setValue)
        
        self.sb_circle_param2.valueChanged.connect(self.update_views)
        self.sl_circle_param2.valueChanged.connect(self.sb_circle_param2.setValue)
        self.sb_circle_param2.valueChanged.connect(self.sl_circle_param2.setValue)
        
        self.sb_local_window.valueChanged.connect(self.update_views)
        self.sl_local_window.valueChanged.connect(self.sb_local_window.setValue)
        self.sb_local_window.valueChanged.connect(self.sl_local_window.setValue)
        
        self.sb_texture_thr.valueChanged.connect(self.update_views)
        self.sl_texture_thr.valueChanged.connect(self.sb_texture_thr.setValue)
        self.sb_texture_thr.valueChanged.connect(self.sl_texture_thr.setValue)
        
        self.sb_texture_window.valueChanged.connect(self.update_views)
        self.sl_texture_window.valueChanged.connect(self.sb_texture_window.setValue)
        self.sb_texture_window.valueChanged.connect(self.sl_texture_window.setValue)
        
        self.sb_seed_x.valueChanged.connect(self.update_views)
        self.sl_seed_x.valueChanged.connect(self.sb_seed_x.setValue)
        self.sb_seed_x.valueChanged.connect(self.sl_seed_x.setValue)
        
        self.sb_seed_y.valueChanged.connect(self.update_views)
        self.sl_seed_y.valueChanged.connect(self.sb_seed_y.setValue)
        self.sb_seed_y.valueChanged.connect(self.sl_seed_y.setValue)
        
        # Tab change handler for lazy loading
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Bind sliders to spin boxes
        self._bind_int_slider(self.sl_thr, self.sb_thr)
        self._bind_int_slider(self.sl_gk, self.sb_gk, step=2, make_odd=True)
        self._bind_scaled_slider(self.sl_gs, self.ds_gs, scale=10.0)  # 1..500 -> 0.1..50.0
        self._bind_scaled_slider(self.sl_la, self.ds_la, scale=10.0)  # 0..50  -> 0.0..5.0
        self._bind_int_slider(self.sl_lk, self.sb_lk, step=2, make_odd=True)
        self._bind_int_slider(self.sl_sk, self.sb_sk, step=2, make_odd=True)
        self._bind_int_slider(self.sl_dx, self.sb_dx)
        self._bind_int_slider(self.sl_dy, self.sb_dy)
        self._bind_int_slider(self.sl_ang, self.ds_ang, is_double=True)

        # Histogram button
        self.btn_hist = QPushButton("Show Histogram")
        self.btn_hist.clicked.connect(self._show_hist_dialog)
        form.addRow("", self.btn_hist)

        # Initial visibility
        self._update_params_visibility()

    def _bind_int_slider(self, slider: QSlider, spin: QWidget, step: int = 1, make_odd: bool = False, is_double: bool = False):
        def on_slider(val: int):
            v = val
            if make_odd:
                if v % 2 == 0:
                    v = v + 1 if v < slider.maximum() else v - 1
            if is_double:
                # QDoubleSpinBox expects float
                spin.setValue(float(v))
            else:
                spin.setValue(int(v))
        def on_spin(val):
            v = int(val)
            if make_odd and v % 2 == 0:
                v += 1
            if slider.value() != v:
                slider.setValue(v)
        slider.valueChanged.connect(on_slider)
        if isinstance(spin, QSpinBox):
            spin.valueChanged.connect(on_spin)
        elif isinstance(spin, QDoubleSpinBox):
            spin.valueChanged.connect(lambda fv: on_spin(int(round(fv))))

    def _bind_scaled_slider(self, slider: QSlider, dspin: QDoubleSpinBox, scale: float):
        # slider int -> double spin with scale factor
        def on_slider(val: int):
            v = round(val / scale, 2)
            if abs(dspin.value() - v) > 1e-6:
                dspin.setValue(v)
        def on_spin(fv: float):
            v = int(round(fv * scale))
            if slider.value() != v:
                slider.setValue(v)
        slider.valueChanged.connect(on_slider)
        dspin.valueChanged.connect(on_spin)

    # Removed old plot tab methods; histogram shows in a dialog now

    def _set_row_visible(self, label: QLabel, field_widget: QWidget, visible: bool):
        label.setVisible(visible)
        field_widget.setVisible(visible)

    def _update_hough_parameters(self):
        """Update Hough line parameters based on adaptive mode"""
        if self._bgr_image is None:
            return
        
        if self.cb_hough_adaptive.isChecked():
            # Calculate adaptive parameters
            gray_hsv = self._gray_hsv if self._gray_hsv is not None else to_gray_hsv_value(self._bgr_image)
            from .image_ops import _calculate_adaptive_hough_params
            adaptive_params = _calculate_adaptive_hough_params(gray_hsv)
            
            # Update UI with adaptive values
            self.sb_hough_thr.setValue(adaptive_params['line_threshold'])
            self.sb_hough_min_len.setValue(adaptive_params['line_min_length'])
            self.sb_hough_max_gap.setValue(adaptive_params['line_max_gap'])
        
        self.update_views()
    
    def _update_circle_parameters(self):
        """Update Hough circle parameters based on adaptive mode"""
        if self._bgr_image is None:
            return
        
        if self.cb_circle_adaptive.isChecked():
            # Calculate adaptive parameters
            gray_hsv = self._gray_hsv if self._gray_hsv is not None else to_gray_hsv_value(self._bgr_image)
            from .image_ops import _calculate_adaptive_hough_params
            adaptive_params = _calculate_adaptive_hough_params(gray_hsv)
            
            # Update UI with adaptive values
            self.sb_circle_min_dist.setValue(adaptive_params['circle_min_dist'])
            self.sb_circle_param1.setValue(adaptive_params['circle_param1'])
            self.sb_circle_param2.setValue(adaptive_params['circle_param2'])
        
        self.update_views()

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
        # Lab 2 parameters
        for widget in self._row_hough_widget:
            widget.setVisible(False)
        for widget in self._row_circle_widget:
            widget.setVisible(False)
        self._set_row_visible(self.lbl_local, self._row_local_widget, False)
        for widget in self._row_texture_widget:
            widget.setVisible(False)

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
        elif name.startswith("Hough Lines"):
            for widget in self._row_hough_widget:
                widget.setVisible(True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Hough Circles"):
            for widget in self._row_circle_widget:
                widget.setVisible(True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Local Statistics"):
            self._set_row_visible(self.lbl_local, self._row_local_widget, True)
            self.btn_hist.setVisible(True)
        elif name.startswith("Texture Segmentation"):
            for widget in self._row_texture_widget:
                widget.setVisible(True)
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
            self.label_hough_lines, self.label_hough_circles,
            self.label_local_mean, self.label_local_var, self.label_local_skew, self.label_local_kurt,
            self.label_texture_seg,
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
            ("hough_lines", self.label_hough_lines),
            ("hough_circles", self.label_hough_circles),
            ("local_mean", self.label_local_mean),
            ("local_var", self.label_local_var),
            ("local_skew", self.label_local_skew),
            ("local_kurt", self.label_local_kurt),
            ("texture_seg", self.label_texture_seg),
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
        
        # Update adaptive parameters for new image
        self._update_hough_parameters()
        self._update_circle_parameters()
        
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

        # Lab 2: Feature Detection - проверяем активную вкладку
        current_tab_name = self.tabs.tabText(self.tabs.currentIndex())
        
        # Если активная вкладка - Lab 2, вычисляем её
        if current_tab_name.startswith("Hough Lines"):
            self._compute_hough_lines(gray_hsv)
        elif current_tab_name.startswith("Hough Circles"):
            self._compute_hough_circles(gray_hsv)
        elif current_tab_name.startswith("Local Statistics"):
            self._compute_local_statistics(gray_hsv)
        elif current_tab_name.startswith("Texture Segmentation"):
            self._compute_texture_segmentation(gray_hsv)
        else:
            # Для неактивных Lab 2 вкладок показываем заглушки
            self._show_lab2_placeholders()

    def _show_lab2_placeholders(self):
        """Show placeholder images for Lab 2 tabs"""
        if self._bgr_image is None:
            return
        
        h, w = self._bgr_image.shape[:2]
        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
        placeholder[:] = [50, 50, 50]  # Dark gray
        
        # Add text overlay
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Click to compute"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(placeholder, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
        
        self.label_hough_lines.set_cv_image(placeholder)
        self.label_hough_circles.set_cv_image(placeholder)
        self.label_local_mean.set_cv_image(placeholder)
        self.label_local_var.set_cv_image(placeholder)
        self.label_local_skew.set_cv_image(placeholder)
        self.label_local_kurt.set_cv_image(placeholder)
        self.label_texture_seg.set_cv_image(placeholder)

    def _on_tab_changed(self, index):
        """Handle tab change for lazy loading of Lab 2 features"""
        if self._bgr_image is None:
            return
        
        tab_name = self.tabs.tabText(index)
        gray_hsv = self._gray_hsv if self._gray_hsv is not None else to_gray_hsv_value(self._bgr_image)
        
        # Check if this is a Lab 2 tab that needs computation
        if tab_name.startswith("Hough Lines"):
            self._compute_hough_lines(gray_hsv)
        elif tab_name.startswith("Hough Circles"):
            self._compute_hough_circles(gray_hsv)
        elif tab_name.startswith("Local Statistics"):
            self._compute_local_statistics(gray_hsv)
        elif tab_name.startswith("Texture Segmentation"):
            self._compute_texture_segmentation(gray_hsv)

    def _compute_hough_lines(self, gray_hsv):
        """Compute Hough lines when tab is accessed"""
        adaptive = bool(self.cb_hough_adaptive.isChecked()) if hasattr(self, 'cb_hough_adaptive') else True
        hough_thr = int(self.sb_hough_thr.value()) if hasattr(self, 'sb_hough_thr') else 100
        hough_min_len = int(self.sb_hough_min_len.value()) if hasattr(self, 'sb_hough_min_len') else 50
        hough_max_gap = int(self.sb_hough_max_gap.value()) if hasattr(self, 'sb_hough_max_gap') else 10
        hough_lines_img, _ = hough_lines(gray_hsv, threshold=hough_thr, 
                                       min_line_length=hough_min_len, max_line_gap=hough_max_gap,
                                       adaptive=adaptive)
        self.label_hough_lines.set_cv_image(hough_lines_img)
        self._tab_image_map["Hough Lines"] = hough_lines_img

    def _compute_hough_circles(self, gray_hsv):
        """Compute Hough circles when tab is accessed"""
        adaptive = bool(self.cb_circle_adaptive.isChecked()) if hasattr(self, 'cb_circle_adaptive') else True
        circle_dp = float(self.ds_circle_dp.value()) if hasattr(self, 'ds_circle_dp') else 1.0
        circle_min_dist = int(self.sb_circle_min_dist.value()) if hasattr(self, 'sb_circle_min_dist') else 50
        circle_param1 = int(self.sb_circle_param1.value()) if hasattr(self, 'sb_circle_param1') else 50
        circle_param2 = int(self.sb_circle_param2.value()) if hasattr(self, 'sb_circle_param2') else 30
        hough_circles_img, _ = hough_circles(gray_hsv, dp=circle_dp, min_dist=circle_min_dist,
                                           param1=circle_param1, param2=circle_param2, adaptive=adaptive)
        self.label_hough_circles.set_cv_image(hough_circles_img)
        self._tab_image_map["Hough Circles"] = hough_circles_img

    def _compute_local_statistics(self, gray_hsv):
        """Compute local statistics when tab is accessed"""
        local_window = int(self.sb_local_window.value()) if hasattr(self, 'sb_local_window') else 15
        # Limit window size to prevent hanging
        local_window = min(local_window, 31)  # Max 31x31 window
        
        try:
            local_mean, local_var, local_skew, local_kurt = local_statistics(gray_hsv, window_size=local_window)
            self.label_local_mean.set_cv_image(local_mean)
            self._tab_image_map["Local Statistics — Mean"] = local_mean
            self.label_local_var.set_cv_image(local_var)
            self._tab_image_map["Local Statistics — Variance"] = local_var
            self.label_local_skew.set_cv_image(local_skew)
            self._tab_image_map["Local Statistics — Skewness"] = local_skew
            self.label_local_kurt.set_cv_image(local_kurt)
            self._tab_image_map["Local Statistics — Kurtosis"] = local_kurt
        except Exception as e:
            # If computation fails, show error
            h, w = gray_hsv.shape[:2]
            error_img = np.zeros((h, w, 3), dtype=np.uint8)
            self.label_local_mean.set_cv_image(error_img)
            self.label_local_var.set_cv_image(error_img)
            self.label_local_skew.set_cv_image(error_img)
            self.label_local_kurt.set_cv_image(error_img)

    def _compute_texture_segmentation(self, gray_hsv):
        """Compute texture segmentation when tab is accessed"""
        texture_thr = int(self.sb_texture_thr.value()) if hasattr(self, 'sb_texture_thr') else 30
        texture_window = int(self.sb_texture_window.value()) if hasattr(self, 'sb_texture_window') else 15
        seed_x = int(self.sb_seed_x.value()) if hasattr(self, 'sb_seed_x') else 100
        seed_y = int(self.sb_seed_y.value()) if hasattr(self, 'sb_seed_y') else 100
        
        # Ensure seed point is within image bounds
        h, w = gray_hsv.shape[:2]
        seed_x = max(0, min(w-1, seed_x))
        seed_y = max(0, min(h-1, seed_y))
        
        try:
            texture_seg_img = texture_segmentation(gray_hsv, seed_point=(seed_y, seed_x), 
                                                 threshold=texture_thr, window_size=texture_window)
            self.label_texture_seg.set_cv_image(texture_seg_img)
            self._tab_image_map["Texture Segmentation"] = texture_seg_img
        except Exception as e:
            # If segmentation fails, show error message
            error_img = np.zeros((h, w, 3), dtype=np.uint8)
            self.label_texture_seg.set_cv_image(error_img)
            self._tab_image_map["Texture Segmentation"] = error_img

    def _update_lab2_features(self, gray_hsv):
        """Update Lab 2 features only when needed"""
        # Hough Lines
        adaptive_lines = bool(self.cb_hough_adaptive.isChecked()) if hasattr(self, 'cb_hough_adaptive') else True
        hough_thr = int(self.sb_hough_thr.value()) if hasattr(self, 'sb_hough_thr') else 100
        hough_min_len = int(self.sb_hough_min_len.value()) if hasattr(self, 'sb_hough_min_len') else 50
        hough_max_gap = int(self.sb_hough_max_gap.value()) if hasattr(self, 'sb_hough_max_gap') else 10
        hough_lines_img, _ = hough_lines(gray_hsv, threshold=hough_thr, 
                                       min_line_length=hough_min_len, max_line_gap=hough_max_gap,
                                       adaptive=adaptive_lines)
        self.label_hough_lines.set_cv_image(hough_lines_img)
        self._tab_image_map["Hough Lines"] = hough_lines_img

        # Hough Circles
        adaptive_circles = bool(self.cb_circle_adaptive.isChecked()) if hasattr(self, 'cb_circle_adaptive') else True
        circle_dp = float(self.ds_circle_dp.value()) if hasattr(self, 'ds_circle_dp') else 1.0
        circle_min_dist = int(self.sb_circle_min_dist.value()) if hasattr(self, 'sb_circle_min_dist') else 50
        circle_param1 = int(self.sb_circle_param1.value()) if hasattr(self, 'sb_circle_param1') else 50
        circle_param2 = int(self.sb_circle_param2.value()) if hasattr(self, 'sb_circle_param2') else 30
        hough_circles_img, _ = hough_circles(gray_hsv, dp=circle_dp, min_dist=circle_min_dist,
                                           param1=circle_param1, param2=circle_param2, adaptive=adaptive_circles)
        self.label_hough_circles.set_cv_image(hough_circles_img)
        self._tab_image_map["Hough Circles"] = hough_circles_img

        # Local Statistics
        local_window = int(self.sb_local_window.value()) if hasattr(self, 'sb_local_window') else 15
        local_mean, local_var, local_skew, local_kurt = local_statistics(gray_hsv, window_size=local_window)
        self.label_local_mean.set_cv_image(local_mean)
        self._tab_image_map["Local Statistics — Mean"] = local_mean
        self.label_local_var.set_cv_image(local_var)
        self._tab_image_map["Local Statistics — Variance"] = local_var
        self.label_local_skew.set_cv_image(local_skew)
        self._tab_image_map["Local Statistics — Skewness"] = local_skew
        self.label_local_kurt.set_cv_image(local_kurt)
        self._tab_image_map["Local Statistics — Kurtosis"] = local_kurt

        # Texture Segmentation
        texture_thr = int(self.sb_texture_thr.value()) if hasattr(self, 'sb_texture_thr') else 30
        texture_window = int(self.sb_texture_window.value()) if hasattr(self, 'sb_texture_window') else 15
        seed_x = int(self.sb_seed_x.value()) if hasattr(self, 'sb_seed_x') else 100
        seed_y = int(self.sb_seed_y.value()) if hasattr(self, 'sb_seed_y') else 100
        
        # Ensure seed point is within image bounds
        h, w = gray_hsv.shape[:2]
        seed_x = max(0, min(w-1, seed_x))
        seed_y = max(0, min(h-1, seed_y))
        
        try:
            texture_seg_img = texture_segmentation(gray_hsv, seed_point=(seed_y, seed_x), 
                                                 threshold=texture_thr, window_size=texture_window)
            self.label_texture_seg.set_cv_image(texture_seg_img)
            self._tab_image_map["Texture Segmentation"] = texture_seg_img
        except Exception as e:
            # If segmentation fails, show error message
            error_img = np.zeros((h, w, 3), dtype=np.uint8)
            self.label_texture_seg.set_cv_image(error_img)
            self._tab_image_map["Texture Segmentation"] = error_img

def run():
    app = QApplication(sys.argv)
    # Modern look
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run()
