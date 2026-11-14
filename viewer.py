"""
viewer.py - "Dancing with the SARs": interactive NISAR change viewer.

Run:   python viewer.py                      (opens the last-used or demo results, or asks)
       python viewer.py out\\overview         (opens a results folder made by process_pair.py)

What you can do:
  * Switch layers: colour composite, the two echo channels, and two change maps.
  * SWIPE: drag the yellow line to compare BEFORE (left) and AFTER (right) on the exact same pixels.
  * Hover: see location, backscatter on both dates and the change at the pixel under the mouse.
  * Weather tab: temperature, rain and snow between the passes (from fetch_weather.py).
  * File > Process new pair...: pick (or drag in) two GCOV .h5 files + an area, and the app builds the maps.
"""
import concurrent.futures
import contextlib
import io
import json
import re
import sys
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PIL import Image
from pyproj import Transformer

import process_pair as pp   # same stretch / colour code as the command-line tool

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

LAYERS = [
    "Colour composite (all channels)",
    "Like-polarised echo (horizontal → horizontal)",
    "Cross-polarised echo (horizontal → vertical)",
    "Change in like-polarised echo",
    "Change in cross-polarised echo",
]
OVERVIEW_BOX = (24.5964, 47.1959, 26.2897, 48.1388)   # lon_min lat_min lon_max lat_max
DETAIL_BOX = (25.30, 47.84, 25.46, 47.95)


# --------------------------------------------------------------------------- data
OV_MAX = 4_000_000        # pixels in the overview picture that is shown when zoomed out
DETAIL_MAX = 9_000_000    # largest area drawn at full resolution (deep zoom)
SAMPLE_MAX = 2_500_000    # pixels used for the statistics on huge scenes


def reduce_blocks(a, f):
    """Average f x f blocks (NaN-aware, float32, in row chunks so memory stays small)."""
    nr, nc = a.shape[0] // f, a.shape[1] // f
    out = np.empty((nr, nc), np.float32)
    step = max(1, 1_500_000 // max(1, nc * f * f))
    for r0 in range(0, nr, step):
        r1 = min(nr, r0 + step)
        blk = np.asarray(a[r0 * f:r1 * f, :nc * f], dtype=np.float32).reshape(r1 - r0, f, nc, f)
        fin = np.isfinite(blk)
        sm = np.where(fin, blk, 0).sum(axis=(1, 3))
        n = fin.sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r0:r1] = np.where(n >= max(1, f * f // 2), sm / np.maximum(n, 1), np.nan)
    return out


class Results:
    """One folder written by process_pair.py: info.json + stack.npz (+ optional weather.json).

    Small scenes are used as they are. Huge scenes (more than OV_MAX pixels) get a reduced overview
    (self.ov, f x f averaged) for zooming out; the full-resolution arrays (self.a) stay in memory as
    float16 and are only used for the area you are looking at, for hover values and for statistics.
    """

    def __init__(self, folder, progress=None):
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "info.json").read_text())
        i = self.info
        self.nr, self.nc = i["shape_rows_cols"]
        self.pix = i["pixel_size_m"]
        self.f = 1 if self.nr * self.nc <= OV_MAX else int(np.ceil(np.sqrt(self.nr * self.nc / OV_MAX)))
        self.nr_o, self.nc_o = self.nr // self.f, self.nc // self.f
        self.a, self.ov, self._vals = {}, {}, {}
        z = np.load(self.folder / "stack.npz")
        for k in z.files:
            if progress:
                progress(k)
            arr = z[k]
            if k == "valid":
                self.valid = arr.astype(bool)
                self.ov["valid"] = self.valid if self.f == 1 else (reduce_blocks(arr.astype(np.float32), self.f) >= 0.5)
                continue
            if self.f == 1:
                self.a[k] = self.ov[k] = arr
            else:
                self.ov[k] = reduce_blocks(arr, self.f)
                self.a[k] = arr.astype(np.float16)
            del arr
        self.x_min = i["utm_extent"]["x_min"]
        self.y_max = i["utm_extent"]["y_max"]
        self.to_ll = Transformer.from_crs(i["epsg"], 4326, always_xy=True)
        self.weather = self._find_weather()

    def vals(self, key):
        """Valid values of one array as a flat float32 vector (sub-sampled on huge scenes), for statistics."""
        if key not in self._vals:
            a = self.a[key]
            st = 1 if a.size <= SAMPLE_MAX else int(np.ceil(np.sqrt(a.size / SAMPLE_MAX)))
            x = np.asarray(a[::st, ::st], dtype=np.float32)[self.valid[::st, ::st]]
            self._vals[key] = x[np.isfinite(x)]
        return self._vals[key]

    def _find_weather(self):
        for p in (self.folder / "weather.json", self.folder.parent / "weather.json", Path.cwd() / "out" / "weather.json"):
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except Exception:
                    return None
        return None

    def lonlat(self, row, col):
        x = self.x_min + (col + 0.5) * self.pix
        y = self.y_max - (row + 0.5) * self.pix
        return self.to_ll.transform(x, y)

    def layer_rgba(self, layer, which, clip, region=None):
        """RGBA uint8 image for a layer; which = 'before' | 'after' (ignored for change layers).

        region=None gives the overview picture; region=(row0, row1, col0, col1) in full-resolution pixels
        gives just that part at full resolution."""
        r = self.info["display_range_db"]
        if region is None:
            get = lambda k: self.ov[k]
            valid = self.ov["valid"]
        else:
            r0, r1, c0, c1 = region
            get = lambda k: self.a[k][r0:r1, c0:c1].astype(np.float32)
            valid = self.valid[r0:r1, c0:c1]
        if layer in (3, 4):
            rgb = pp.change_rgb(get("change_hh" if layer == 3 else "change_hv"), clip)
        else:
            hh, hv = get(f"hh_{which}"), get(f"hv_{which}")
            if layer == 0:
                rgb = np.dstack([pp.u8(hh, *r["hh"]), pp.u8(hv, *r["hv"]), pp.u8(hh - hv, *r["hh_minus_hv"])])
            elif layer == 1:
                g = pp.u8(hh, *r["hh"]); rgb = np.dstack([g, g, g])
            else:
                g = pp.u8(hv, *r["hv"]); rgb = np.dstack([g, g, g])
        return np.dstack([rgb, valid.astype(np.uint8) * 255])


# --------------------------------------------------------------------------- image widget
class ImageView(pg.GraphicsLayoutWidget):
    hovered = QtCore.Signal(int, int)
    resized = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setBackground("#1b1b1f")
        self.vb = self.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.img = pg.ImageItem()                       # overview (or the whole picture on small scenes)
        self.vb.addItem(self.img)
        self.img_d = pg.ImageItem(); self.img_d.setZValue(5); self.img_d.setVisible(False)   # full-resolution crop
        self.vb.addItem(self.img_d)
        # thin marker line; it is moved only by the slider under the picture
        self.line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffd54a", width=3))
        self.line.setZValue(50)                          # above the full-detail layer (z=5) and everything else
        self.vb.addItem(self.line)
        self.frame = pg.PlotCurveItem(pen=pg.mkPen("#9aa3b5", width=2))   # outline of the picture = what a saved image covers
        self.frame.setZValue(40)
        self.vb.addItem(self.frame)
        self._proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self._moved)

    def set_frame(self, nc, nr):
        self.frame.setData([0, nc, nc, 0, 0], [0, 0, nr, nr, 0])

    def _moved(self, evt):
        p = self.vb.mapSceneToView(evt[0])
        self.hovered.emit(int(np.floor(p.x())), int(np.floor(p.y())))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.resized.emit()


# --------------------------------------------------------------------------- processing dialog
class Worker(QtCore.QThread):
    log = QtCore.Signal(str)
    done = QtCore.Signal(bool, str)

    def __init__(self, argv, outdir):
        super().__init__()
        self.argv, self.outdir = argv, outdir

    def run(self):
        sig = self.log

        class Writer:
            def write(self, t):
                if t:
                    sig.emit(t)

            def flush(self):
                pass

        ok = True
        try:
            with contextlib.redirect_stdout(Writer()):
                pp.main(self.argv)
        except SystemExit as e:
            ok = e.code in (0, None)
        except Exception:
            self.log.emit(traceback.format_exc())
            ok = False
        self.done.emit(ok, self.outdir)


def file_date_key(path):
    """Start time inside a NISAR filename (first 8-digit date + time), used to tell BEFORE from AFTER."""
    m = re.search(r"_(\d{8}T\d{6})_", Path(path).name)
    return m.group(1) if m else Path(path).name


class ProcessDialog(QtWidgets.QDialog):
    finished_ok = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Process a new pair of NISAR GCOV files")
        self.setAcceptDrops(True)
        self.resize(760, 560)
        self.worker = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "Drag two GCOV .h5 files here (same track and frame!) or browse. The earlier date becomes BEFORE."))
        form = QtWidgets.QFormLayout()
        self.before = QtWidgets.QLineEdit(); self.after = QtWidgets.QLineEdit()
        form.addRow("BEFORE .h5", self._with_browse(self.before, "file"))
        form.addRow("AFTER .h5", self._with_browse(self.after, "file"))
        self.box = []
        boxrow = QtWidgets.QHBoxLayout()
        for label, val in zip(("lon min", "lat min", "lon max", "lat max"), OVERVIEW_BOX):
            sp = QtWidgets.QDoubleSpinBox(); sp.setDecimals(4); sp.setRange(-180, 180); sp.setValue(val)
            self.box.append(sp)
            boxrow.addWidget(QtWidgets.QLabel(label)); boxrow.addWidget(sp)
        form.addRow("Area (degrees)", boxrow)
        presets = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Preset: overview (whole box)"); b2 = QtWidgets.QPushButton("Preset: detail (small box)")
        b1.clicked.connect(lambda: self._preset(OVERVIEW_BOX, 2, "overview"))
        b2.clicked.connect(lambda: self._preset(DETAIL_BOX, 3, "detail"))
        presets.addWidget(b1); presets.addWidget(b2); presets.addStretch()
        form.addRow("", presets)
        self.block = QtWidgets.QSpinBox(); self.block.setRange(1, 20); self.block.setValue(4)
        self.block.setToolTip("Average NxN pixels. 20 m x N. Higher = cleaner and faster, lower = sharper but noisier.")
        self.name = QtWidgets.QLineEdit("overview")
        self.outroot = QtWidgets.QLineEdit(str(Path.cwd() / "out"))
        form.addRow("Block size (px)", self.block)
        form.addRow("Result name", self.name)
        form.addRow("Output folder", self._with_browse(self.outroot, "dir"))
        lay.addLayout(form)
        self.logbox = QtWidgets.QPlainTextEdit(); self.logbox.setReadOnly(True)
        self.logbox.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        lay.addWidget(self.logbox, 1)
        self.run_btn = QtWidgets.QPushButton("Process")
        self.run_btn.clicked.connect(self.run)
        lay.addWidget(self.run_btn)

    def _with_browse(self, edit, kind):
        w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        btn = QtWidgets.QPushButton("Browse...")

        def pick():
            if kind == "file":
                p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose GCOV file", "", "NISAR HDF5 (*.h5)")
            else:
                p = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose folder")
            if p:
                edit.setText(p)

        btn.clicked.connect(pick)
        h.addWidget(edit, 1); h.addWidget(btn)
        return w

    def _preset(self, box, block, name):
        for sp, v in zip(self.box, box):
            sp.setValue(v)
        self.block.setValue(block); self.name.setText(name)

    # drag & drop of .h5 files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.toLocalFile().lower().endswith(".h5")]
        files.sort(key=file_date_key)
        if len(files) >= 2:
            self.before.setText(files[0]); self.after.setText(files[-1])
        elif len(files) == 1:
            (self.before if not self.before.text() else self.after).setText(files[0])

    def run(self):
        b, a = self.before.text().strip(), self.after.text().strip()
        if not (b and a):
            self.logbox.appendPlainText("Pick both files first.")
            return
        name, root = self.name.text().strip() or "run", self.outroot.text().strip() or "out"
        lon0, lat0, lon1, lat1 = [sp.value() for sp in self.box]
        argv = [b, a, "--bbox", str(lon0), str(lat0), str(lon1), str(lat1),
                "--block", str(self.block.value()), "--name", name, "--out", root]
        # how big will the result be? (upper bound: the box may extend past the satellite strip)
        bsz = self.block.value()
        lat_c = (lat0 + lat1) / 2
        w_px = (lon1 - lon0) * 111320 * np.cos(np.radians(lat_c)) / (20 * bsz)
        h_px = (lat1 - lat0) * 110574 / (20 * bsz)
        total = w_px * h_px
        if total > 25e6:
            n = int(np.ceil(np.sqrt(total * bsz * bsz / 9e6)))
            mb = QtWidgets.QMessageBox(self)
            mb.setWindowTitle("This is a very large result")
            mb.setIcon(QtWidgets.QMessageBox.Warning)
            mb.setText(f"Block size {bsz} on this area gives about {total / 1e6:.0f} million pixels "
                       f"({w_px:,.0f} x {h_px:,.0f}).")
            mb.setInformativeText("Processing needs several GB of memory and a few minutes, and the file is large. The viewer copes "
                                  "with it (zoomed out you see a faster averaged overview, zoomed in you see every pixel), but a "
                                  f"block size of {n} gives about {total / (n / bsz) ** 2 / 1e6:.0f} million pixels and is much lighter.")
            use = mb.addButton(f"Use block size {n}", QtWidgets.QMessageBox.AcceptRole)
            go = mb.addButton("Continue anyway", QtWidgets.QMessageBox.ActionRole)
            mb.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            mb.exec()
            if mb.clickedButton() is use:
                total = total * (bsz / n) ** 2
                self.block.setValue(n)
                argv[argv.index("--block") + 1] = str(n)
            elif mb.clickedButton() is not go:
                return
        self.logbox.clear()
        self.run_btn.setEnabled(False)
        if total > 9e6:
            argv.append("--no-png")       # the viewer does not use the PNG previews; skipping them saves minutes
        px = (self.box[2].value() - self.box[0].value()) * 111320 * 0.67 / (20 * self.block.value())
        self.logbox.appendPlainText(f"Working... about {px:,.0f} px wide at this block size. Big areas with block size 1 can take "
                                    f"several minutes and a lot of memory. This window closes by itself when it is done.\n")
        self._t0 = QtCore.QElapsedTimer(); self._t0.start()
        self._tick = QtCore.QTimer(self); self._tick.setInterval(1000)
        self._tick.timeout.connect(lambda: self.run_btn.setText(
            f"Processing...  {self._t0.elapsed() // 60000}:{self._t0.elapsed() // 1000 % 60:02d} elapsed (please wait)"))
        self._tick.start(); self.run_btn.setText("Processing...")
        self.worker = Worker(argv, str(Path(root) / name))
        self.worker.log.connect(lambda t: self.logbox.insertPlainText(t))
        self.worker.done.connect(self._done)
        self.worker.start()

    def _done(self, ok, outdir):
        self._tick.stop(); self.run_btn.setText("Process")
        self.run_btn.setEnabled(True)
        if ok:
            self.logbox.appendPlainText("\nFinished. Opening results...")
            self.finished_ok.emit(outdir)
        else:
            self.logbox.appendPlainText("\nFailed - see the messages above.")
            title, text = self._explain_failure(self.logbox.toPlainText())
            QtWidgets.QMessageBox.warning(self, title, text)

    @staticmethod
    def _explain_failure(log):
        """Plain-language pop-up text for failures the tool recognises; a generic one otherwise."""
        for marker, title, hint in (
            ("CANNOT USE THESE FILES:", "These files cannot be used",
             "Pick two L-band GCOV files (name contains _GCOV_) with frequency A and HH/HV layers."),
            ("NOT COMPARABLE:", "These two files cannot be compared",
             "Pick two files from the same track, frame, pass direction and map grid."),
        ):
            if marker in log:
                lines = []
                for ln in log.split(marker, 1)[1].splitlines()[1:]:
                    if ln.strip().startswith("- "):
                        lines.append(ln.strip()[2:])
                    elif lines:
                        break
                return title, "\n".join("\u2022 " + t for t in lines) + "\n\n" + hint
        return "Processing failed", "Something went wrong while processing. The details are in the log below the progress bar."


# --------------------------------------------------------------------------- text helpers
HANDLE = 26          # slider handle width in px (must match STYLE)
LIKE, CROSS = "Like-polarised", "Cross-polarised"


def nice_date(s):
    d = datetime.strptime(s[:10], "%Y-%m-%d")
    return f"{d.day} {d:%b %Y}"


def short_date(s):
    d = datetime.strptime(s[:10], "%Y-%m-%d")
    return f"{d.day} {d:%b}"


def days_between(a, b):
    return (datetime.strptime(b[:10], "%Y-%m-%d") - datetime.strptime(a[:10], "%Y-%m-%d")).days


def verdict(db):
    a = abs(db)
    if a < 1:
        return "about the same"
    word = "slightly" if a < 3 else "clearly" if a < 6 else "strongly"
    return f"{word} {'darker' if db < 0 else 'brighter'}"


def table(rows, head=None):
    h = ""
    if head:
        h = "<tr>" + "".join(f"<td style='color:#8d93a0'>{c}</td>" for c in head) + "</tr>"
    body = "".join("<tr><td style='color:#8d93a0; padding-right:14px'>%s</td>%s</tr>" % (
        r[0], "".join(f"<td>{c}</td>" for c in r[1:])) for r in rows)
    return f"<table cellspacing='0' cellpadding='3'>{h}{body}</table>"


def swatch(colour):
    return f"<span style='background-color:{colour}'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>"


STYLE = """
QWidget { font-size: 13px; }
QMainWindow, QDialog { background: #1b1c20; }
QLabel#h { font-size: 11px; color: #8d93a0; letter-spacing: 1px; }
QLabel#ct { font-size: 12px; color: #8fb4ff; font-weight: bold; letter-spacing: 1px; }
QFrame#card { background: #25272d; border: 1px solid #343741; border-radius: 8px; }
QFrame#card QLabel { background: transparent; }
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit { background: #2b2d34; border: 1px solid #3b3e49; border-radius: 5px; padding: 5px 8px; }
QComboBox:hover, QDoubleSpinBox:hover { border-color: #5b8ff0; }
QPushButton { background: #33363f; border: 1px solid #454956; border-radius: 5px; padding: 6px 12px; }
QPushButton:hover { background: #3f4350; border-color: #5b8ff0; }
QTabWidget::pane { border: 1px solid #343741; border-radius: 6px; top: -1px; }
QTabBar::tab { background: #25272d; padding: 7px 12px; border: 1px solid #343741; border-bottom: none;
               border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; color: #a9afbd; }
QTabBar::tab:selected { background: #31343d; color: #ffffff; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QPlainTextEdit { background: #1f2024; border: none; padding: 6px; }
QToolButton#helpbig { background: transparent; border: none; border-radius: 20px; padding: 3px; }
QToolButton#helpbig:hover { background: #33415f; }
QWidget#bottombar { background: #1b1b1f; }
QToolBar#ribbon { background: #23252b; border: none; border-bottom: 1px solid #343741; spacing: 4px; padding: 4px 8px; }
QToolBar#ribbon QToolButton { color: #cfd6e6; padding: 4px 10px; border-radius: 6px; font-size: 11px; }
QToolBar#ribbon QToolButton:hover { background: #33415f; }
QToolBar#ribbon QToolButton:checked { background: #2b3a66; }
QToolBar#ribbon::separator { width: 1px; background: #3b3e49; margin: 6px 6px; }
QSlider { background: transparent; }
QSlider::groove:horizontal { height: 8px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3a5aa8, stop:1 #b8862f); border-radius: 4px; }
QSlider::handle:horizontal { background: #ffd54a; width: 26px; margin: -9px 0; border-radius: 13px; border: 2px solid #ffffff; }
QSlider::handle:horizontal:hover { background: #ffffff; }
QSlider:disabled::handle:horizontal { background: #555a66; border-color: #777; }
QStatusBar { color: #a9afbd; }
QToolButton#help { background: transparent; border: none; padding: 2px; margin-right: 4px; }
QToolButton#help:hover { background: #33415f; border-radius: 12px; }
QListWidget { background: #1f2024; border: none; padding: 4px; }
QListWidget::item { padding: 8px 10px; border-radius: 5px; }
QListWidget::item:selected { background: #31405f; color: #ffffff; }
QCheckBox { spacing: 8px; }
QScrollBar:vertical { background: #1f2024; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #4a4e5b; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #5b8ff0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""


def help_icon(size=64, ring="#8fb4ff", fill="#2b3a66"):
    """A round '?' badge drawn with QPainter, so the button shows a real image."""
    pm = QtGui.QPixmap(size, size); pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor(ring), size * 0.06)); p.setBrush(QtGui.QColor(fill))
    m = size * 0.06
    p.drawEllipse(QtCore.QRectF(m, m, size - 2 * m, size - 2 * m))
    f = QtGui.QFont(); f.setBold(True); f.setPixelSize(int(size * 0.62)); p.setFont(f)
    p.setPen(QtGui.QColor("#ffffff")); p.drawText(QtCore.QRectF(0, 0, size, size), QtCore.Qt.AlignCenter, "?")
    p.end()
    return QtGui.QIcon(pm)


def ribbon_icon(kind, size=48):
    """Simple line icons drawn with QPainter (no image files needed)."""
    pm = QtGui.QPixmap(size, size); pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor("#cfd6e6"), size * 0.06, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
    p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush)
    u = size / 48.0
    R = lambda x, y, w, h: QtCore.QRectF(x * u, y * u, w * u, h * u)
    P = lambda x, y: QtCore.QPointF(x * u, y * u)
    if kind == "open":
        p.drawPolygon(QtGui.QPolygonF([P(6, 14), P(19, 14), P(23, 19), P(42, 19), P(42, 38), P(6, 38)]))
    elif kind == "process":
        p.drawEllipse(R(15, 15, 18, 18))
        for k in range(8):
            import math
            a = k * math.pi / 4
            p.drawLine(P(24 + 11 * math.cos(a), 24 + 11 * math.sin(a)), P(24 + 17 * math.cos(a), 24 + 17 * math.sin(a)))
    elif kind == "save":
        p.drawRoundedRect(R(8, 8, 32, 32), 3 * u, 3 * u)
        p.drawRect(R(14, 8, 18, 10)); p.drawRect(R(14, 26, 20, 14))
    elif kind == "fit":
        for (x, y, dx, dy) in ((8, 8, 1, 1), (40, 8, -1, 1), (8, 40, 1, -1), (40, 40, -1, -1)):
            p.drawLine(P(x, y), P(x + 10 * dx, y)); p.drawLine(P(x, y), P(x, y + 10 * dy))
    elif kind == "frame":
        pen.setStyle(QtCore.Qt.DashLine); p.setPen(pen); p.drawRect(R(9, 11, 30, 26))
    elif kind == "terms":
        p.drawRoundedRect(R(11, 7, 26, 34), 3 * u, 3 * u)
        for y in (17, 24, 31):
            p.drawLine(P(17, y), P(31, y))
    elif kind == "help":
        p.drawEllipse(R(7, 7, 34, 34))
        f = QtGui.QFont(); f.setBold(True); f.setPixelSize(int(26 * u)); p.setFont(f)
        p.drawText(R(7, 7, 34, 34), QtCore.Qt.AlignCenter, "?")
    p.end()
    return QtGui.QIcon(pm)


class AutoLabel(QtWidgets.QLabel):
    """Word-wrapped rich-text label that reserves its true wrapped height.

    The height is measured with a separate QTextDocument. (QLabel.heightForWidth includes the label's own
    minimum height, so using it here made the height grow on every layout pass.)
    """

    def __init__(self, html=""):
        super().__init__(html)
        self.setWordWrap(True)
        self.setTextFormat(QtCore.Qt.RichText)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self._busy = False

    def _fit(self):
        if self._busy:
            return
        self._busy = True
        try:
            doc = QtGui.QTextDocument()
            doc.setDefaultFont(self.font())
            doc.setHtml(self.text())
            doc.setTextWidth(max(self.width(), 60))
            h = int(doc.size().height()) + 6
            if abs(h - self.minimumHeight()) > 2:
                self.setMinimumHeight(h)
        finally:
            self._busy = False

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def setText(self, t):
        super().setText(t)
        self._fit()


def heading(text):
    l = QtWidgets.QLabel(text.upper()); l.setObjectName("h")
    return l


def make_card(title, html=None, widget=None, topic=None, help_cb=None):
    f = QtWidgets.QFrame(); f.setObjectName("card")
    lay = QtWidgets.QVBoxLayout(f); lay.setContentsMargins(14, 10, 14, 12); lay.setSpacing(6)
    row = QtWidgets.QHBoxLayout()
    t = QtWidgets.QLabel(title.upper()); t.setObjectName("ct"); t.setWordWrap(True)
    row.addWidget(t, 1)
    if topic and help_cb:
        b = QtWidgets.QToolButton(); b.setObjectName("help"); b.setText("?")
        b.setToolTip("Explain this, with sources"); b.setCursor(QtCore.Qt.PointingHandCursor)
        b.clicked.connect(lambda _=False, tp=topic: help_cb(tp))
        row.addWidget(b)
    lay.addLayout(row)
    if html is not None:
        l = AutoLabel(html)
        l.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        l.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(l)
    if widget is not None:
        lay.addWidget(widget)
    return f


class CardList(QtWidgets.QScrollArea):
    """A scrolling column of titled boxes."""

    def __init__(self):
        super().__init__()
        self.help_cb = None
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        body = QtWidgets.QWidget()
        self.lay = QtWidgets.QVBoxLayout(body); self.lay.setSpacing(10); self.lay.setContentsMargins(6, 8, 10, 8)
        self.lay.addStretch(1)
        self.setWidget(body)

    def clear(self):
        while self.lay.count() > 1:
            it = self.lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def add(self, title, html=None, widget=None, topic=None):
        c = make_card(title, html, widget, topic, self.help_cb)
        self.lay.insertWidget(self.lay.count() - 1, c)
        return c


class QuietPlot(pg.PlotWidget):
    """Plot that never eats the mouse wheel, so scrolling a panel over a chart keeps scrolling the panel."""

    def wheelEvent(self, e):
        e.ignore()


def make_plot(height, xlabel=None, ylabel=None):
    pw = QuietPlot(background="#1f2024")
    pw.setFixedHeight(height)
    pw.setMouseEnabled(False, False); pw.hideButtons(); pw.setMenuEnabled(False)
    pw.showGrid(x=False, y=True, alpha=0.18)
    p = pw.getPlotItem()
    for ax in ("left", "bottom"):
        p.getAxis(ax).setTextPen("#a9afbd"); p.getAxis(ax).setPen("#555a66")
    p.getAxis("left").setWidth(48); p.getAxis("bottom").setHeight(34)
    if xlabel:
        p.setLabel("bottom", xlabel, color="#a9afbd")
    if ylabel:
        p.setLabel("left", ylabel, color="#a9afbd")
    return pw


def legend_html(items):
    return "&nbsp;&nbsp;&nbsp;".join(f"<span style='color:{c}'>■</span> {t}" for c, t in items)


# --------------------------------------------------------------------------- legend (what the picture means)
def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def key_table(rows):
    return ("<table cellspacing='0' cellpadding='3'>" +
            "".join(f"<tr><td style='padding-right:10px' valign='middle'>{sym}</td><td>{txt}</td></tr>" for sym, txt in rows) +
            "</table>")


def swipe_sub(m, b, a):
    if m == 3:
        return f"Swipe: left is the <b>satellite photo</b> (Sentinel-2, 2024), right is the radar of <b>{a}</b>."
    if m == 0:
        return f"Swipe: left is <b>{b}</b>, right is <b>{a}</b>."
    return "Showing <b>" + (b if m == 1 else a) + "</b> only."


def colour_classes(res, which="after"):
    """Share of valid pixels in each colour family of the composite (red=HH, green=HV, blue=HH-HV).

    Uses the same stretch limits as the picture, so the shares match what is on screen."""
    r = res.info["display_range_db"]
    hh, hv = res.vals(f"hh_{which}"), res.vals(f"hv_{which}")
    n = min(hh.size, hv.size)
    hh, hv = hh[:n], hv[:n]
    if n == 0:
        return []
    nz_ = lambda x, lim: np.clip((x - lim[0]) / (lim[1] - lim[0]), 0, 1)
    R, G, B = nz_(hh, r["hh"]), nz_(hv, r["hv"]), nz_(hh - hv, r["hh_minus_hv"])
    left = np.ones(n, bool)
    out = []
    defs = [
        ("dark", "dark: weak echo in both channels", (np.maximum(R, G) < 0.30)),
        ("white", "white / pale pink: all three strong", (np.minimum(np.minimum(R, G), B) > 0.55)),
        ("purple", "purple / magenta: HH strong, HV weaker", (R >= 0.45) & (B >= 0.45) & (G < np.minimum(R, B) - 0.10)),
        ("green", "green / yellow-green: HV strong", (G >= R - 0.05) & (G >= B)),
        ("red", "red / orange: HH strong, ratio low", (R >= G) & (R >= B) & (B < 0.45)),
        ("blue", "blue: HH above HV, both modest", (B > R) & (B > G)),
    ]
    for key, text, cond in defs:
        m = left & cond
        left &= ~m
        if m.any():
            col = [float(np.mean(c[m])) * 255 for c in (R, G, B)]
            out.append((key, text, m.mean() * 100, hexc(col)))
    if left.any():
        out.append(("other", "other mixes", left.mean() * 100, "#808080"))
    return out


def legend_cards(res, L, m, clip, base_on):
    i = res.info
    b, a = nice_date(i["before"]["start"]), nice_date(i["after"]["start"])
    sh = i["median_shift_db"]
    r = i["display_range_db"]
    change = L in (3, 4)
    swipe = (not change) and m in (0, 3)
    cards = []

    rows = []
    if change:
        pts = np.array([[-clip, -clip / 2, 0, clip / 2, clip]])
        cols = pp.change_rgb(pts, clip)[0]
        labels = [f"{clip:g} dB weaker or more", f"about {clip / 2:g} dB weaker", "no change",
                  f"about {clip / 2:g} dB stronger", f"{clip:g} dB stronger or more"]
        rows += [(swatch(hexc(c)), t) for c, t in zip(cols, labels)]
        name = "like-polarised (HH)" if L == 3 else "cross-polarised (HV)"
        title = f"Change in the {name} echo"
        sub = f"<b>{a}</b> minus <b>{b}</b>. Scene-wide shift of {sh['hh' if L == 3 else 'hv']:+.2f} dB removed."
        note = "Stronger or weaker echo, not a diagnosis. Compare with the Weather tab."
    elif L == 0:
        title = "Colour composite"
        sub = swipe_sub(m, b, a)
        which = "before" if m == 1 else "after"
        cls = colour_classes(res, which)
        rows += [(swatch(c), f"{t}: <b>{p:.0f}%</b>") for _, t, p, c in cls]
        note = ("Red = HH, green = HV, blue = HH minus HV (NASA SAR Handbook recipe). Colours add: red + blue = purple. "
                "Shares are for <b>" + (b if which == "before" else a) + "</b>. Meaning of each colour: press ?.")
    else:
        lo, hi = r["hh" if L == 1 else "hv"]
        vals = np.linspace(lo, hi, 5)
        rows += [(swatch(hexc([np.clip((x - lo) / (hi - lo), 0, 1) * 255] * 3)),
                  f"{x:.0f} dB" + (" (weak)" if k == 0 else " (strong)" if k == 4 else "")) for k, x in enumerate(vals)]
        title = "Like-polarised echo (HH)" if L == 1 else "Cross-polarised echo (HV)"
        sub = swipe_sub(m, b, a)
        note = "Brighter means a stronger radar echo."
    rows.append((swatch("#2a2a30"), "no data"))
    if swipe:
        rows.append(("<span style='color:#ffd54a'><b>&#9474;</b></span>", "swipe line (slider below)"))
    if base_on:
        rows.append(("<span style='color:#7fb06f'>&#9638;</span>", "satellite photo: context only"))
    cards.append((title, f"{sub}<br>" + key_table(rows) + f"<br><span style='color:#a9afbd'>{note}</span>", "colours" if L == 0 else "reading"))

    ny, nx = res.nr * res.pix / 1000, res.nc * res.pix / 1000
    clon, clat = res.lonlat(res.nr / 2, res.nc / 2)
    bk = i.get("block", "?")
    cards.append(("Scale and location", table([
        ("Direction", "north is up"),
        ("One pixel", f"{res.pix:.0f} m ({bk}x{bk} original 20 m pixels averaged)"),
        *([("Zoomed out", f"{res.f} x {res.f} averaged; zoom in for every pixel")] if res.f > 1 else []),
        ("Picture", f"{res.nc} x {res.nr} px, about {nx:.0f} x {ny:.0f} km"),
        ("Centre", f"{clat:.3f}° N, {clon:.3f}° E"),
    ]), "method"))

    if change:
        ch = res.vals("change_hh" if L == 3 else "change_hv")
        n = max(ch.size, 1)
        num = table([
            ("Darker by more than 3 dB", f"<b>{(ch < -3).sum() / n * 100:.1f}%</b>"),
            ("Brighter by more than 3 dB", f"<b>{(ch > 3).sum() / n * 100:.1f}%</b>"),
            ("Within ±1 dB", f"<b>{(np.abs(ch) < 1).sum() / n * 100:.1f}%</b>"),
        ])
    else:
        rws = []
        for key, nm in (("hh", LIKE), ("hv", CROSS)):
            if L in (0, 1 if key == "hh" else 2):
                x0, x1 = np.median(res.vals(f"{key}_before")), np.median(res.vals(f"{key}_after"))
                rws.append((f"{nm}, typical", f"<b>{x0:.1f}</b> → <b>{x1:.1f}</b> dB ({x1 - x0:+.1f})"))
        num = table(rws)
    cards.append(("Numbers", num, "decibels"))
    cards.append(("Data quality",
                  "NISAR products carry a maturity level (BETA, PROVISIONAL, VALIDATED). BETA data are not fully calibrated. "
                  "Compare the two dates with each other rather than trusting absolute brightness, and do not mix maturities or "
                  "processing versions (CRID). Stripes and bright blobs can be artifacts. See the guide.", "issues"))
    return cards


class LegendView(QtWidgets.QTextBrowser):
    """The legend as ONE scrolling document (one scroll bar, no nested widgets, so it cannot fight itself)."""

    helpRequested = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setOpenLinks(False)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.document().setDefaultStyleSheet("a { color: #8fb4ff; text-decoration: none; } td { padding: 2px 8px 2px 0; }")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.anchorClicked.connect(lambda u: self.helpRequested.emit(u.toString().replace("help:", "")))

    def set_cards(self, cards):
        keep = self.verticalScrollBar().value()
        out = []
        for title, html, topic in cards:
            link = f" &nbsp;<a href='help:{topic}'>[?]</a>" if topic else ""
            out.append(f"<p style='margin-bottom:2px'><b style='color:#8fb4ff; letter-spacing:1px'>{title.upper()}</b>{link}</p>"
                       f"<div>{html}</div><hr>")
        self.setHtml("".join(out))
        self.verticalScrollBar().setValue(keep)


# --------------------------------------------------------------------------- weather panel
def nz(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def pretty_key(k):
    k = k.replace("_C", " (°C)").replace("_mm", " (mm)").replace("_cm", " (cm)").replace("_", " ")
    return k.capitalize()


def fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else ("n/a" if v is None else str(v))


class WeatherPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Place:")); self.place = QtWidgets.QComboBox(); row.addWidget(self.place, 1)
        lay.addLayout(row)
        self.cards = CardList(); lay.addWidget(self.cards, 1)
        self.place.currentIndexChanged.connect(self.draw)
        self.res = None

    def set_results(self, res):
        self.res = res
        self.place.blockSignals(True); self.place.clear()
        if res.weather:
            self.place.addItems(list(res.weather["places"].keys()))
        self.place.blockSignals(False)
        self.draw()

    def draw(self, *_):
        self.cards.clear()
        res = self.res
        if not (res and res.weather):
            hint = str(res.folder / "info.json") if res else "out\\overview\\info.json"
            self.cards.add("No weather data yet",
                           "Weather helps explain radar changes. Run this once in a terminal, then reopen the results:<br><br>"
                           "<code>python fetch_weather.py --info " + hint + "</code>")
            return
        pl = res.weather["places"].get(self.place.currentText())
        if not pl or not pl.get("daily"):
            return
        d = pl["daily"]; n = len(d); x = np.arange(n)
        tmin = np.array([nz(r["tmin_C"]) for r in d]); tmax = np.array([nz(r["tmax_C"]) for r in d])
        rain = np.array([nz(r["precip_mm"]) for r in d]); snow = np.array([nz(r["snowfall_cm"]) for r in d])
        step = max(2, n // 4)
        ticks = [(k, short_date(d[k]["date"])) for k in range(n) if k % step == 0 or k == n - 1]
        b, a = nice_date(res.info["before"]["start"]), nice_date(res.info["after"]["start"])

        # temperature
        pt = make_plot(190, ylabel="°C")
        p = pt.getPlotItem()
        c1 = p.plot(x, tmax, pen=pg.mkPen("#ff9a4a", width=2), symbol="o", symbolSize=5, symbolBrush="#ff9a4a")
        c2 = p.plot(x, tmin, pen=pg.mkPen("#ffd54a", width=2), symbol="o", symbolSize=5, symbolBrush="#ffd54a")
        p.addItem(pg.FillBetweenItem(c1, c2, brush=pg.mkBrush(255, 160, 60, 40)))
        p.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("#7fd0ff", style=QtCore.Qt.DashLine)))
        for xi in (0, n - 1):
            p.addItem(pg.InfiniteLine(xi, angle=90, pen=pg.mkPen("#ffffff", width=1, style=QtCore.Qt.DotLine)))
        p.getAxis("bottom").setTicks([ticks]); p.setXRange(-0.5, n - 0.5, padding=0)
        lab = QtWidgets.QLabel(legend_html([("#ff9a4a", "highest of the day"), ("#ffd54a", "lowest of the day"),
                                             ("#7fd0ff", "freezing point (0 °C)")]))
        lab.setStyleSheet("font-size:12px"); lab.setWordWrap(True)
        box = QtWidgets.QWidget(); bl = QtWidgets.QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0); bl.addWidget(lab); bl.addWidget(pt)
        self.cards.add("Temperature between the passes", widget=box)

        # rain and snow
        pr = make_plot(170, ylabel="mm  /  cm")
        q = pr.getPlotItem()
        q.addItem(pg.BarGraphItem(x=x, height=np.nan_to_num(rain), width=0.6, brush="#4a86e8"))
        if np.nansum(snow) > 0:
            q.addItem(pg.BarGraphItem(x=x, height=np.nan_to_num(snow), width=0.28, brush="#eef3ff"))
        for xi in (0, n - 1):
            q.addItem(pg.InfiniteLine(xi, angle=90, pen=pg.mkPen("#ffffff", width=1, style=QtCore.Qt.DotLine)))
        q.getAxis("bottom").setTicks([ticks]); q.setXRange(-0.5, n - 0.5, padding=0)
        lab2 = QtWidgets.QLabel(legend_html([("#4a86e8", "rain (millimetres)"), ("#eef3ff", "snow (centimetres)")]) +
                                f"<br><span style='color:#8d93a0'>Dotted lines: the satellite passes ({b} and {a}).</span>")
        lab2.setStyleSheet("font-size:12px"); lab2.setWordWrap(True)
        box2 = QtWidgets.QWidget(); b2 = QtWidgets.QVBoxLayout(box2); b2.setContentsMargins(0, 0, 0, 0); b2.addWidget(lab2); b2.addWidget(pr)
        self.cards.add("Rain and snow between the passes", widget=box2)

        # what it suggests
        tp, ts = np.nansum(rain), np.nansum(snow)
        bits = [f"Between the passes: <b>{tp:.1f} mm</b> of precipitation, <b>{ts:.1f} cm</b> of snowfall, "
                f"temperatures from <b>{np.nanmin(tmin):.1f}</b> to <b>{np.nanmax(tmax):.1f} °C</b>."]
        if np.nanmin(tmin) < 0:
            bits.append("• Below-freezing nights: frozen soil or frozen canopy can lower the radar echo, especially the cross-polarised one.")
        if ts > 0.5:
            bits.append("• Snow fell: snow cover can darken or brighten the image depending on whether it is dry or wet.")
        if tp > 10:
            bits.append("• Notable rain: wetter soil often raises the like-polarised echo, especially over bare fields.")
        bits.append("<span style='color:#8d93a0'>These are hints, not proof.</span>")
        self.cards.add("What this might mean for the radar", "<br>".join(bits), topic="weather")

        # conditions at the passes
        pb, pa = pl.get("before") or {}, pl.get("after") or {}
        keys = list(dict.fromkeys(list(pb) + list(pa)))
        if keys:
            self.cards.add("Conditions just before each pass",
                           table([(pretty_key(k), fmt(pb.get(k)), fmt(pa.get(k))) for k in keys], head=["", b, a]))

        # daily table
        rows = [(r["date"], fmt(nz(r["tmin_C"])), fmt(nz(r["tmax_C"])), fmt(nz(r["precip_mm"])),
                 fmt(nz(r["snowfall_cm"])), fmt(r.get("snow_depth_cm_end"))) for r in d]
        self.cards.add("Day by day", table(rows, head=["Date", "Low °C", "High °C", "Rain mm", "Snow cm", "Snow depth cm"]))
        self.cards.add("Where this comes from",
                       f"Open-Meteo historical archive (ERA5 reanalysis, a grid of about 25 km). Cell around {pl['lat']:.3f}° N, {pl['lon']:.3f}° E "
                       f"with model elevation {pl.get('elevation_m')} m: not a weather station, and mountain weather can differ a lot from the cell average. "
                       "Cite Open-Meteo and ERA5 (Hersbach et al., 2023); see the ? for sources.", topic="weather")


# --------------------------------------------------------------------------- statistics panel
class StatsPanel(CardList):
    def set_results(self, res):
        self.clear()
        edges = np.linspace(-10, 10, 81); cen = (edges[:-1] + edges[1:]) / 2
        pw = make_plot(220, xlabel="Local change, after minus before (dB)", ylabel="% of pixels")
        p = pw.getPlotItem()
        rows = []
        for key, colour, nm in (("change_hv", "#6fd08c", CROSS), ("change_hh", "#d98cff", LIKE)):
            vv = res.vals(key)
            if not vv.size:
                continue
            h, _ = np.histogram(np.clip(vv, -10, 10), bins=edges)
            p.plot(cen, h / h.sum() * 100, pen=pg.mkPen(colour, width=2))
            rows.append((f"<span style='color:{colour}'>■</span> {nm}", f"{(vv < -3).mean() * 100:.1f}%",
                         f"{(vv > 3).mean() * 100:.1f}%", f"{(np.abs(vv) < 1).mean() * 100:.0f}%"))
        p.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen("#ffffff", style=QtCore.Qt.DashLine)))
        p.setXRange(-10, 10, padding=0)
        lab = QtWidgets.QLabel(legend_html([("#6fd08c", "cross-polarised echo"), ("#d98cff", "like-polarised echo")]))
        lab.setStyleSheet("font-size:12px"); lab.setWordWrap(True)
        box = QtWidgets.QWidget(); bl = QtWidgets.QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0); bl.addWidget(lab); bl.addWidget(pw)
        self.add("How much of the scene changed?", widget=box)
        self.add("The numbers", table(rows, head=["", "Darker >3 dB", "Brighter >3 dB", "Within ±1 dB"]), topic="decibels")
        self.add("How to read this chart",
                 "Each point shows what share of pixels changed by that many decibels. A <b>tall narrow spike at 0</b> means most of the "
                 "scene is stable. <b>Fat tails</b> to the left (darker) or right (brighter) are real change, or noise: bigger blocks give "
                 "cleaner tails.<br><br>The scene-wide median shift was removed first, so this describes <i>local</i> change only.", topic="method")


# --------------------------------------------------------------------------- guide & sources (pop-up window)
def badge(text, bg, fg):
    return f"<span style='background-color:{bg}; color:{fg}; font-size:11px'>&nbsp;<b>{text}</b>&nbsp;</span>"


DOC = badge("FROM DOCUMENTATION", "#1f5f3a", "#c9f2d6")
APP = badge("THIS APP'S METHOD", "#5a4a14", "#ffe9a8")
INTERP = badge("INTERPRETATION (NOT FROM THE SOURCES)", "#5a2a2a", "#ffcaca")
MATH = badge("ARITHMETIC", "#2a3a5a", "#cfe0ff")

U = dict(
    mission="https://science.nasa.gov/mission/nisar/",
    eop="https://www.eoportal.org/satellite-missions/nisar",
    clouds="https://www.nasa.gov/missions/nisar/nasa-isro-satellite-captures-pacific-northwest-through-clouds",
    whatsar="https://www.earthdata.nasa.gov/learn/what-is-sar",
    asfsar="https://hyp3-docs.asf.alaska.edu/guides/introduction_to_sar/",
    gcov="https://nisar-docs.asf.alaska.edu/gcov/",
    l2gcov="https://nisar-docs.asf.alaska.edu/l2-gcov",
    gcov_spacing="https://hyp3-docs.asf.alaska.edu/nisar-docs/gcov",
    prod="https://nisar-docs.asf.alaska.edu/products-overview/",
    spec="https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/DOCS/NISAR_D-102274_RevE_NASA_SDS_Product_Specification_L2_GCOV_Nov8_2024_w-sigs.pdf",
    issues="https://nisar-docs.asf.alaska.edu/product-known-issues/",
    avail="https://nisar-docs.asf.alaska.edu/availability-overview/",
    prov="https://nisar-docs.asf.alaska.edu/provisional-known-issues/",
    beta="https://catalog.data.gov/dataset/nisar-beta-geocoded-polarimetric-covariance-product-version-1",
    om="https://open-meteo.com/en/docs/historical-weather-api",
    eox="https://eox.at/2025/03/sentinel-2-cloudless-2024/",
    eoxlic="https://cloudless.eox.at/license-non-commercial",
    s2="https://s2maps.eu",
    arset="https://www.earthdata.nasa.gov/s3fs-public/2026-07/arset-2026-nisar-part2.pdf",
)


def src(*pairs):
    return ("<p style='color:#8d93a0'><b>Sources</b><br>" +
            "<br>".join(f"<a href='{u}'>{t}</a>" for t, u in pairs) + "</p>")


HELP_CSS = ("a { color: #8fb4ff; } h2 { color: #ffffff; } h3 { color: #8fb4ff; } "
            "td { padding: 3px 10px 3px 0; } li { margin-bottom: 4px; }")

U.update(
    handbook="https://www.earthdata.nasa.gov/s3fs-public/2025-04/SARHB_CH3_Content.pdf",
    handbook_doi="https://doi.org/10.25966/hbm1-ej07",
    projections="https://nisar-docs.asf.alaska.edu/projections",
    quick="https://nisar.jpl.nasa.gov/mission/quick-facts/",
    radars="https://nisar.jpl.nasa.gov/mission/observatory/radars/",
    park="https://www.mdpi.com/2072-4292/7/12/15874",
    hardaway="https://www.asprs.org/wp-content/uploads/pers/1982journal/mar/1982_mar_399-404.pdf",
    thirion="https://doi.org/10.3390/rs12101632",
    akhavan="https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1065/",
    sweden="https://doi.org/10.1109/TGRS.2009.2023906",
    era5doc="https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation",
    era5="https://doi.org/10.24381/cds.adbb2d47",
    zip="https://doi.org/10.5281/zenodo.7970649",
)

REFS = [
    ("NASA/JPL, NISAR quick facts and radar page", "quick", "radars"),
    ("eoPortal, NISAR mission", "eop", None),
    ("ASF DAAC, NISAR Data User Guide: GCOV", "gcov", None),
    ("ASF DAAC, NISAR data products overview (maturity levels, product levels)", "prod", None),
    ("ASF DAAC, NISAR Level 2 projections", "projections", None),
    ("ASF DAAC, known issues of pre-calibration (BETA) products", "issues", None),
    ("ASF DAAC, known issues of PROVISIONAL products", "prov", None),
    ("ASF DAAC, NISAR data availability and release timeline", "avail", None),
    ("Shiroma (2024), NISAR Level-2 GCOV product specification, JPL D-102274 Rev E", "spec", None),
    ("ASF, Introduction to SAR", "asfsar", None),
    ("Flores et al. (2019), The SAR Handbook, chapter 3 (NASA Earthdata)", "handbook", "handbook_doi"),
    ("Park et al. (2015), freeze/thaw backscatter, ALOS PALSAR, Remote Sensing 7(12)", "park", None),
    ("Hardaway et al. (1982), Cardinal effect on Seasat images of urban areas, PE&RS", "hardaway", None),
    ("Thirion-Lefevre et al. (2020), orientation angle and material in PolSAR urban images, Remote Sensing 12(10)", "thirion", None),
    ("Open-Meteo, Historical Weather API", "om", None),
    ("Hersbach et al. (2023), ERA5 hourly data on single levels (Copernicus C3S)", "era5", None),
    ("EOX, Sentinel-2 cloudless 2024 and licence", "eox", "eoxlic"),
    ("Santoro, Fransson, Eriksson, Magnusson, Ulander and Olsson (2009), Signatures of ALOS PALSAR L-band backscatter in Swedish forest, IEEE TGRS 47(12), 4001-4019", "sweden", None),
    ("NASA, NISAR captures the Pacific Northwest through clouds", "clouds", None),
    ("ECMWF, ERA5: data documentation (Copernicus Knowledge Base)", "era5doc", None),
]


def ref(*nums):
    return "<sup style='color:#8fb4ff'>[" + ",".join(str(n) for n in nums) + "]</sup>"


def refs_html():
    out = "<ol>"
    for t, k, k2 in REFS:
        out += f"<li>{t}. <a href='{U[k]}'>{U[k]}</a>" + (f" and <a href='{U[k2]}'>{U[k2]}</a>" if k2 else "") + "</li>"
    return out + "</ol>"


HELP_TOPICS = [
    ("start", "Start here", f"""
<h2>Guide and sources</h2>
<p>The Legend tab gives only the essentials for reading the current picture. This guide holds the detail. Each statement is labelled with where it comes from, and numbers like {ref(6)} point to the <b>References</b> section at the end:</p>
<p>{DOC} &nbsp;Stated on an official NISAR, ASF, NASA or peer-reviewed source.</p>
<p>{APP} &nbsp;A choice made by this program. Not a standard.</p>
<p>{INTERP} &nbsp;Reasoning from the documented facts. A hypothesis to test, not a result.</p>
<p>{MATH} &nbsp;Plain arithmetic.</p>
<p>Sources accessed 14 November 2026. NISAR documentation is still changing during calibration; where a link and this guide disagree, the link is authoritative.</p>
<p>Reading order: <i>Terms</i> (short definitions), <i>Colours of the composite</i>, <i>What different surfaces do at L-band</i>, <i>Data maturity and known issues</i>.</p>
"""),
    ("terms", "Terms (quick glossary)", f"""
<h2>Terms</h2>
<p>Short definitions. Follow the link for the full explanation.</p>
<table cellspacing='0' cellpadding='4'><tr><td valign='top' style='padding-right:12px'><b>SAR</b></td><td>Synthetic aperture radar: an active imaging radar that builds a sharp image by combining echoes along the flight path. <a href='{U['whatsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>L-band</b></td><td>Radar band of about 23 to 24 cm wavelength (1.25 GHz for NISAR). Penetrates vegetation more than shorter bands. <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Backscatter</b></td><td>The part of the transmitted radar energy that returns to the antenna. Brighter pixels mean stronger return. <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>dB (decibel)</b></td><td>Logarithmic scale: 10 x log10 of power. +3 dB is about double, -3 dB about half, +10 dB ten times. <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Polarisation</b></td><td>Orientation of the radar wave. Two letters: transmit then receive (H horizontal, V vertical). <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>HH</b></td><td>Horizontal sent, horizontal received. A co-polarised channel; responds to surfaces and double bounce. <a href='{U['handbook']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>HV</b></td><td>Horizontal sent, vertical received. A cross-polarised channel; most sensitive to volume scattering such as canopy. <a href='{U['handbook']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>VV</b></td><td>Vertical sent, vertical received. Co-polarised; shows rough-surface scattering best. Absent from dual-pol HH/HV products. <a href='{U['handbook']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Co-pol / cross-pol</b></td><td>Same transmit and receive orientation (HH, VV) versus different orientation (HV, VH). <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Surface scattering</b></td><td>Reflection from a ground surface; bright if rough, dark if smooth. <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Volume scattering</b></td><td>Multiple bounces inside a layer such as leaves and branches. Raises cross-pol return. <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Double bounce</b></td><td>Reflection ground-to-wall or ground-to-trunk and back to the sensor. Very bright, typical of buildings. <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Speckle</b></td><td>Grainy noise present in all SAR images, caused by interference of echoes inside a pixel. <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Multi-looking</b></td><td>Averaging neighbouring samples to reduce speckle at the cost of resolution. <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Layover / shadow</b></td><td>Slope-related distortions: front slopes folded onto each other (layover); steep back slopes hidden (shadow). <a href='{U['asfsar']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>RTC</b></td><td>Radiometric terrain correction: normalises backscatter for slope and local area using a DEM. <a href='{U['gcov']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>DEM</b></td><td>Digital elevation model: the terrain heights used for RTC (Copernicus 30 m / 90 m for NISAR GCOV). <a href='{U['gcov']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Gamma-0 / sigma-0</b></td><td>Two normalisation conventions for backscatter. GCOV stores gamma-0 and provides a factor to convert to sigma-0. <a href='{U['gcov']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>GCOV</b></td><td>Geocoded polarimetric covariance: NISAR Level-2 backscatter product on a map grid. <a href='{U['gcov']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>RSLC</b></td><td>Range-Doppler single-look complex: the Level-1 focused image from which GCOV is derived. <a href='{U['prod']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Track / frame</b></td><td>Track: the repeating orbit path. Frame: a fixed-length segment along it. Same track and frame give the same geometry. <a href='{U['gcov']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>Ascending / descending</b></td><td>Satellite moving north or south during the pass. Geometry differs, so do not compare across them. <a href='{U['gcov']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>CRID</b></td><td>Composite Release ID: version of the processing workflow, found in the file name. <a href='{U['gcov']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>BETA / PROVISIONAL / VALIDATED</b></td><td>NISAR data maturity levels, from not fully calibrated to fully calibrated and validated. <a href='{U['prod']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>RFI</b></td><td>Radio-frequency interference: bright blobs or bands from other transmitters. <a href='{U['issues']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>UTM / EPSG</b></td><td>Metre-based map projection by zone, and the numeric code that identifies a coordinate system. <a href='{U['projections']}'>more</a></td></tr><tr><td valign='top' style='padding-right:12px'><b>ERA5</b></td><td>Global weather reanalysis (about 25 km grid, hourly) used by the Weather tab. <a href='{U['era5']}'>more</a></td></tr></table>
"""),
    ("mission", "The satellite and mission", f"""
<h2>The satellite and mission</h2>
<p>{DOC}</p>
<ul>
<li>NISAR is a joint NASA and ISRO radar mission. NASA's JPL built the <b>L-band</b> radar (about 24 cm wavelength, 1.25 GHz) and ISRO the S-band radar {ref(1,2)}. This app reads L-band only.</li>
<li>Sun-synchronous orbit at about 747 km, 98.4 degrees inclination, exact repeat every <b>12 days</b>, descending/ascending nodal crossings near 6 AM and 6 PM. The antenna looks to one side (south-pointing) over a swath about 242 km wide {ref(1,2)}.</li>
<li>Launched 30 July 2025 from Sriharikota {ref(8)}.</li>
<li>Radar is an active sensor using microwaves that pass through clouds, and it needs no sunlight {ref(19)}.</li>
<li>Longer wavelength penetrates deeper: ASF notes that L-band has the most penetration of the common bands, returning from below the canopy where X-band returns from the canopy top {ref(10)}. The SAR Handbook says L-band interacts with large branches and trunks {ref(11)}.</li>
</ul>
<p>{INTERP} Two passes of the same track (12 days apart or a multiple) share the same viewing geometry, which is what makes subtracting them meaningful.</p>
{src(("NASA/JPL NISAR quick facts", U['quick']), ("NASA/JPL NISAR radars", U['radars']), ("eoPortal", U['eop']), ("ASF availability overview", U['avail']), ("ASF: Introduction to SAR", U['asfsar']))}
"""),
    ("product", "The data product (GCOV)", f"""
<h2>The data product: GCOV</h2>
<p>{DOC}</p>
<ul>
<li><b>GCOV</b>, Geocoded Polarimetric Covariance, is a Level-2 product derived from the Level-1 RSLC (range-Doppler single-look complex) image {ref(3,4,9)}.</li>
<li>The diagonal terms (HHHH, HVHV, ...) are real numbers describing backscatter intensity per polarisation. They are radiometrically terrain corrected, "gamma-0 power values", and ASF says they can be used like NRB or RTC products {ref(3)}. The covariance terms are power, not decibels; the app converts them.</li>
<li>Terrain correction: an area-based RTC, using the Copernicus 30 m / 90 m DEM, normalises the backscatter. It is then geocoded to the output grid with <b>area-based adaptive multi-looking</b>: the averaging window and number of looks vary with topography and geometry. Extra layers: mask (describes the averaging ensemble), numberOfLooks, rtcGammaToSigmaFactor {ref(3,9)}.</li>
<li>Grid: a UTM zone for acquisitions 0 to 60 degrees north (EPSG 32601-32660) {ref(5)}. Pixel spacing follows range bandwidth: 5 MHz gives 80 m, 20 MHz gives 20 m, 40 MHz gives 10 m {ref(3)}. Most land products are 10 or 20 m for frequency A, and frequency B is 80 m {ref(3)}.</li>
<li>Off-diagonal terms exist only for quad-pol acquisitions; dual-pol users must derive them from GSLC {ref(3)}. A dual-pol product (HH and HV) therefore has no VV channel.</li>
<li>The intended uses are biomass, soil moisture, disturbance detection, inundation mapping and crop-area delineation {ref(4)}.</li>
<li>File name fields: instrument (L/S), level, processing type, cycle (12 days each), relative orbit (track), A/D pass direction, frame, bandwidth mode and polarisation codes (for example DH = HH/HV dual), start and end time (UTC), CRID (processing workflow version), orbit accuracy, coverage (full/partial) {ref(3)}.</li>
</ul>
<p>{DOC} Mask layer: an unsigned byte on the same grid as the data, describing the averaging ensemble behind each pixel. <b>0</b> = invalid or partially focused (the ensemble contains at least one invalid or partially focused sample); <b>1 to 5</b> = valid, all averaged samples fully focused, the value being the sub-swath number most samples came from; <b>255</b> = fill value, outside the image bounds {ref(9)}, section 4.3.3.</p>
<p>{APP} Only pixels with mask 1 to 5 (and finite, positive power) are used.</p>
{src(("ASF GCOV", U['gcov']), ("ASF products overview", U['prod']), ("ASF projections", U['projections']), ("Product specification (PDF)", U['spec']))}
"""),
    ("polarisation", "Polarisation (HH and HV)", f"""
<h2>Polarisation: what HH and HV are</h2>
<p>{DOC}</p>
<ul>
<li>The two letters are transmit then receive. <b>HH</b> = horizontal sent, horizontal received. <b>HV</b> = horizontal sent, vertical received. Same-orientation is <b>co-polarised</b> ("like-polarised" in this app); different is <b>cross-polarised</b> {ref(10)}.</li>
<li>Co-polarised returns are typically stronger for <b>surface scattering</b>; cross-polarised returns are associated with <b>volume scattering</b> {ref(11)}.</li>
<li>Three basic mechanisms {ref(10,11)}: surface (a reflecting surface, bright if rough, dark if smooth), volume (waves bouncing among leaves and branches), double bounce (ground then wall or trunk, back to the sensor; ASF calls it the brightest return, found mostly in urban areas).</li>
<li>The SAR Handbook links high HH backscatter to double bounce in flooded and degraded forests {ref(11)}.</li>
<li>Surface scattering is shown best by VV. Without a VV channel, HH is a mixed channel (surface and double bounce) {ref(11)}.</li>
</ul>
<p>{INTERP} In plain terms: HV is the best channel here for tree canopy and tall crops, HH for buildings, trunks and ground. Neither is a pure measure.</p>
{src(("ASF: Introduction to SAR", U['asfsar']), ("SAR Handbook ch. 3", U['handbook']))}
"""),
    ("colours", "Colours of the composite", f"""
<h2>Colours of the composite picture</h2>
<p>{DOC} The SAR Handbook's recipe for dual-polarised data: <b>co-pol to red, cross-pol to green, and the co/cross ratio to blue</b> {ref(11)}. That is exactly what this app does.</p>
<p>{APP} Concretely, each pixel is built from three numbers, each stretched between fixed display limits (listed in the info.json written with each result):</p>
<ul>
<li><b>Red</b> = HH level in dB.</li>
<li><b>Green</b> = HV level in dB.</li>
<li><b>Blue</b> = HH minus HV in dB. This is the co/cross ratio, because a difference of decibels is a ratio of powers.</li>
</ul>
<p>{MATH} Screen colours mix additively: red + blue = <b>purple/magenta</b>; red + green = yellow; green + blue = cyan; all three high = white; all low = black. So <b>purple means: HH strong, HV comparatively weak (blue high), so the ratio is high.</b> Yellow-green means HV strong and HH not much stronger (ratio low, blue off). Dark blue means both weak but HH still above HV.</p>
<p>{DOC} A NASA/JPL description of a NISAR composite of this kind {ref(19)}: colours represent combinations and intensities of polarisations; <b>magenta</b> dots come from strong reflection off flat surfaces such as roads and buildings; <b>yellow-green</b> marks forests and wetlands; <b>dark blue</b> marks relatively smooth surfaces, both water and exposed mountaintops; <b>purple</b> squares with precise right angles inside lighter-green vegetation are human-made, most likely forest thinning or regrowth after thinning. Yellow patches can come from land cover, soil moisture and surface geometry.</p>
<p>{INTERP} What this tends to mean, based on the mechanisms above {ref(11)}:</p>
<ul>
<li><b>Purple/magenta</b>: echo dominated by like-polarised return with little volume scattering. Typical of bare, ploughed or short-vegetation ground and of rough surfaces; also possible on slopes facing the radar.</li>
<li><b>Green to yellow-green</b>: strong cross-polarised return, volume scattering, such as forest canopy. The Handbook says forests appear in shades of green, brighter with more biomass {ref(11)}.</li>
<li><b>White / pale pink</b>: everything strong. In towns this is common, as buildings give strong double bounce {ref(10)}; also steep rocky or rough areas.</li>
<li><b>Red / orange</b>: strong HH, moderate HV, ratio low to middling.</li>
<li><b>Dark / blue-black</b>: weak echo in both. Smooth water is the classic case (ASF: water pixels remain very dark {ref(10)}); also radar shadow and very smooth surfaces.</li>
</ul>
<p>These are tendencies, not classifications. Moisture, slope and noise all move the colours. The Legend tab shows the measured share of each colour group in the loaded scene.</p>
{src(("SAR Handbook ch. 3 (Sec. 3.2.2)", U['handbook']), ("ASF: Introduction to SAR", U['asfsar']))}
"""),
    ("surfaces", "What different surfaces do at L-band", f"""
<h2>What different surfaces do at L-band</h2>
<p>{DOC} unless marked otherwise.</p>
<ul>
<li><b>Forest.</b> Cross-pol is high in dense forest because volume scattering dominates; cross-polarised data show increased brightness where forests are more mature and higher in biomass {ref(11)}. L-band penetrates deep into canopies {ref(11)}.</li>
<li><b>Bare soil.</b> At L-band bare surfaces look smoother and darker than at shorter wavelengths {ref(11)}. Rough ground (for example slash after clearing) can raise backscatter {ref(11)}.</li>
<li><b>Soil and vegetation moisture.</b> Increased moisture tends to increase backscatter {ref(11)}. The Handbook adds that moisture variations are short in time and space and look more random than real disturbance {ref(11)}.</li>
<li><b>Open water.</b> Calm water is very dark because it reflects energy away from the sensor; wind roughening can make it bright {ref(11)}.</li>
<li><b>Urban.</b> Double bounce gives the brightest returns {ref(10)}. How bright depends on how streets and walls are oriented relative to the radar's look direction, known as the cardinal effect {ref(13)}, and also on wall material {ref(14)}. So two similar towns can look different.</li>
<li><b>Frozen ground and snow.</b> ASF notes L-band penetrates snow and ice deeply {ref(10)}. Backscatter drops between thawed and frozen conditions: in airborne L-band over Alaskan taiga, about 3 to 5 dB lower in co-pol and about 6 dB lower in cross-pol when frozen {ref(12)}. Another study found backscatter in frozen conditions in Swedish forest could be similar to unfrozen, depending on temperature and snow history {ref(18)}. So the sign and size depend on the site.</li>
</ul>
<p>{INTERP} Common candidate causes of change between two passes: frost or thaw, snowfall, rain-soaked ground, ploughing, harvest and logging. The Weather tab shows temperature and precipitation between the passes to help rule candidates in or out. The radar alone cannot identify the cause.</p>
{src(("SAR Handbook ch. 3", U['handbook']), ("ASF: Introduction to SAR", U['asfsar']), ("Park et al. 2015", U['park']), ("Hardaway et al. 1982", U['hardaway']), ("Thirion-Lefevre et al. 2020", U['thirion']))}
"""),
    ("decibels", "Decibels and speckle", f"""
<h2>Decibels, speckle and averaging</h2>
<p>{DOC}</p>
<ul>
<li>The dB scale is 10 times the base-10 logarithm of the power. It brightens dark pixels so dark features are easier to separate, and ASF warns it is not suited to every statistical analysis {ref(10)}.</li>
<li><b>Speckle</b> gives a grainy, salt-and-pepper look and occurs in all SAR scenes; multi-looking averages it down at the cost of resolution {ref(10)}. GCOV is already multi-looked adaptively {ref(3)}.</li>
</ul>
<p>{MATH} +3 dB is about twice the power, -3 dB about half, +10 dB ten times, -10 dB one tenth. A change of 6 dB is about a factor of four.</p>
<p>{APP} Blocks are averaged in linear power, then converted to dB. The overview picture averages in dB for display only.</p>
{src(("ASF: Introduction to SAR", U['asfsar']))}
"""),
    ("terrain", "Terrain, layover and shadow", f"""
<h2>Terrain, layover and shadow</h2>
<p>{DOC}</p>
<ul>
<li>A side-looking radar makes slopes facing it appear brighter. GCOV applies radiometric terrain correction to reduce the bias {ref(3)}.</li>
<li><b>Foreshortening</b>: a steep front slope appears as a narrow bright band. <b>Layover</b>: front-slope data cannot be separated out. <b>Shadow</b>: a steep back slope cannot be imaged and offers no information. In RTC products such pixels are No Data and are not interpolated {ref(10)}.</li>
<li>Terrain correction depends on geolocation. ASF shows an example where a small DEM-related offset in one image gives an RTC error {ref(6)}.</li>
</ul>
<p>{INTERP} In rugged terrain some relief remains visible after correction. Comparing two passes of the same track removes most of it from change maps, but not necessarily all.</p>
{src(("ASF: Introduction to SAR", U['asfsar']), ("ASF GCOV", U['gcov']), ("ASF known issues", U['issues']))}
"""),
    ("issues", "Data maturity and known issues", f"""
<h2>Data maturity and known issues</h2>
<p>{DOC}</p>
<ul>
<li>ASF defines three maturity levels. <b>BETA</b>: the first releases (January and February 2026), "not fully calibrated, and exhibit a number of known issues". <b>PROVISIONAL</b>: released July 2026, acquisitions from 17 June 2026, fully calibrated and partially validated. <b>VALIDATED</b>: released 6 November 2026 for acquisitions from 21 October 2026 {ref(4)}.</li>
<li>BETA products cover earlier acquisitions (October 2025 to January 2026) in some areas, and BETA CRIDs are lower than P05023 {ref(7,8)}.</li>
<li>ASF says to use caution when mixing maturities, because differences can come from processing, not from the landscape {ref(4,8)}.</li>
</ul>
<p>{INTERP} Comparisons are most reliable between products of the same maturity level and the same CRID. Both are readable from the file names; differing values suggest a processing change, not a landscape change.</p>
<p>Known BETA issues relevant to the GCOV backscatter {ref(6)}:</p>
<ul>
<li><b>Valid-data mask misalignment</b>: it exposes artifacts at the edges of geocoded products including GCOV. ASF gives no offset size.</li>
<li><b>Radiometric banding</b>: bright and dark bands in the along-track direction from an incompletely removed cross-track antenna pattern, most visible over uniform surfaces. Frequency B shows it more than frequency A.</li>
<li><b>Streaks</b>: parallel streaks in very low backscatter areas (lakes, sand), most noticeable in HV.</li>
<li><b>RFI</b>: bright blobs or sharp range-direction bands, more evident in low-reflectivity areas, especially HV.</li>
<li><b>Geolocation</b>: granules are within 5 m, which can show in the highest-resolution RTC images; quick-look KML browse products can be mislocated by kilometres.</li>
</ul>
<p>{INTERP} Practical consequence: stripes running one way across the whole picture, bright blobs in dark regions, and odd edges are more likely instrument or processing artifacts than land change. Long smooth bands tend to cancel when you subtract the two dates, but do not count on it.</p>
{src(("ASF BETA known issues", U['issues']), ("ASF PROVISIONAL known issues", U['prov']), ("ASF products overview", U['prod']), ("ASF availability", U['avail']))}
"""),
    ("method", "What this app does to the data", f"""
<h2>What this app does to the data</h2>
<p>{APP} Everything below is this program's design, not NISAR's.</p>
<ol>
<li><b>Safety check.</b> Both files must share track, frame, pass direction and grid. Otherwise processing stops.</li>
<li><b>Window.</b> Only the chosen area is read.</li>
<li><b>Valid pixels.</b> Mask values 1 to 5 only (see the product topic).</li>
<li><b>Block averaging</b> in linear power in N x N blocks (a block needs at least half its pixels valid).</li>
<li><b>Decibels</b>, then <b>change = after minus before</b>.</li>
<li><b>Median shift removed</b>: the scene-wide median change is subtracted, so the map shows local change. It assumes most of the scene did not really change; a truly widespread change is partly hidden.</li>
<li><b>Words</b>: within 1 dB is "about the same", 1 to 3 "slightly", 3 to 6 "clearly", above 6 "strongly". These cut-offs are this app's choice.</li>
<li><b>Overview and zoom</b>: for large scenes the zoomed-out picture averages f x f pixels (in dB, display only); zooming in shows the full-resolution pixels. Hover values always come from full resolution.</li>
<li><b>Not done</b>: edge trimming for the mask artifact; incidence-angle or ionospheric correction.</li>
</ol>
"""),
    ("area", "Map grid and coordinates", f"""
<h2>Map grid and coordinates</h2>
<p>{DOC}</p>
<ul>
<li>Level-2 products are on a map grid: a UTM zone for acquisitions between 0 and 60 degrees north (EPSG 32601-32660), south counterparts to 60 degrees south, and polar stereographic beyond {ref(5)}.</li>
<li>Pixels are square in map metres, with north up and east right {ref(5)}.</li>
</ul>
<p>{APP} The viewer converts pixel positions to latitude and longitude with the grid's EPSG code. Coordinates in the hover readout refer to the pixel centre. The processing window can extend beyond the data strip; areas without data are drawn as no data.</p>
{src(("ASF NISAR Level 2 projections", U['projections']))}
"""),
    ("reading", "How to read each layer", f"""
<a name="layer0"></a><h2>Colour composite</h2>
<p>See the <b>Colours of the composite</b> topic for the full recipe and what each colour tends to mean. Red = HH, green = HV, blue = HH minus HV {ref(11)}.</p>
<a name="layer1"></a><h2>Like-polarised echo (HH)</h2>
<p>{DOC} Co-pol returns are typically stronger for surface scattering, and HH is also strong for double bounce {ref(11)}. Grey scale: dark is weak echo, bright is strong, between the display limits in the Legend.</p>
<p>{INTERP} Buildings, trunks and rough ground are bright; smooth water and shadow are dark.</p>
<a name="layer2"></a><h2>Cross-polarised echo (HV)</h2>
<p>{DOC} Volume scattering is most sensitive to cross-pol {ref(11)}. Forests look brighter with more biomass {ref(11)}.</p>
<p>{INTERP} This is the main channel for canopy and vegetation change.</p>
<a name="layer3"></a><h2>Change maps (both)</h2>
<p>{APP} Change = after minus before in dB, scene median shift removed. Blue = weaker echo than before, red = stronger, white = unchanged. Layer 4 is the same for HV, layer 3 for HH.</p>
<p>{INTERP} Large, evenly coloured patches with sharp edges often come from human activity (ploughing, harvest, logging, construction). Broad smooth gradients, for example with altitude, point more to weather. Isolated dots are probably speckle. These are rules of thumb: the Handbook only says that moisture variation looks more random than real disturbance {ref(11)}. The NISAR docs list soil moisture, disturbance, inundation and crops as intended uses {ref(4)}.</p>
<a name="layer4"></a><h2>Change in HV vs HH</h2>
<p>{INTERP} Change in HV (vegetation volume) while HH stays flat suggests canopy or crop changes. Change in HH while HV stays flat suggests surface or moisture changes. A change in both by a similar amount is more likely radiometric or moisture-related. This is reasoning from the mechanisms above, not a documented test.</p>
"""),
    ("weather", "Weather data", f"""
<h2>Weather data</h2>
<p>{DOC}</p>
<ul>
<li>The Weather tab uses the <b>Open-Meteo historical archive</b>, based on <b>ERA5</b> reanalysis on a 0.25 degree grid (roughly 25 km), hourly {ref(15,16)}.</li>
<li>Reanalysis combines a model forecast with observations (data assimilation); ERA5 provides hourly estimates. The native resolution is about 31 km, and the Climate Data Store copy is regridded to 0.25 degrees {ref(16,20)}. Values therefore describe a grid cell, not an exact pixel.</li>
<li>The most recent days are released as ERA5T, about 5 days behind real time, and can differ slightly from the final release issued roughly two to three months later {ref(16,20)}.</li>
<li>Credit: Open-Meteo (Zippenfenig 2023) and ERA5 (Hersbach et al. 2023), "Generated using Copernicus Climate Change Service information" {ref(15,16)}.</li>
</ul>
<p>{INTERP} In mountains one cell mixes valley and ridge, so real temperature and snow at your pixel can differ a lot.</p>
{src(("Open-Meteo Historical Weather API", U['om']), ("ERA5 (C3S)", U['era5']), ("ECMWF ERA5 data documentation", U['era5doc']))}
"""),
    ("basemap", "Satellite photo underlay", f"""
<h2>Satellite photo underlay</h2>
<p>{DOC}</p>
<ul>
<li><b>Sentinel-2 cloudless 2024</b> by EOX IT Services GmbH, a yearly cloud-free mosaic of Copernicus Sentinel-2 imagery {ref(17)}. Free for non-commercial use with attribution (shown under the picture); a commercial use needs a licence {ref(17)}.</li>
<li>Tiles stop at web-mercator zoom 14.</li>
</ul>
<p>{APP} Tiles are downloaded on demand, cached in a folder next to the app, and resampled to the radar grid. Offline, the photo is not shown.</p>
<p>{INTERP} The mosaic shows some moment in 2024, not November 2025; fields and snow can differ. Use it for orientation only.</p>
{src(("EOX Sentinel-2 cloudless 2024", U['eox']), ("EOX licence", U['eoxlic']))}
"""),
    ("refs", "References", f"""
<h2>References</h2>
{refs_html()}

"""),
]



class HelpWindow(QtWidgets.QDialog):
    """Movable, resizable, non-modal guide with sources."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guide and sources")
        self.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint, True)
        self.setModal(False)
        self.setSizeGripEnabled(True)
        self.resize(980, 700)
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("Guide and sources"); t.setStyleSheet("font-size:18px; font-weight:bold")
        self.ontop = QtWidgets.QCheckBox("Keep this window on top")
        self.ontop.toggled.connect(self._ontop)
        top.addWidget(t); top.addStretch(1); top.addWidget(self.ontop)
        lay.addLayout(top)
        split = QtWidgets.QSplitter()
        self.list = QtWidgets.QListWidget()
        for key, title, _ in HELP_TOPICS:
            self.list.addItem(title)
        self.list.setMinimumWidth(230); self.list.setMaximumWidth(340)
        self.view = QtWidgets.QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.document().setDefaultStyleSheet(HELP_CSS)
        split.addWidget(self.list); split.addWidget(self.view)
        split.setStretchFactor(1, 1); split.setSizes([260, 720])
        lay.addWidget(split, 1)
        self.list.currentRowChanged.connect(self._row)
        self.keys = [k for k, _, _ in HELP_TOPICS]
        self.list.setCurrentRow(0)

    def _ontop(self, on):
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, on)
        self.show()

    def _row(self, i):
        if 0 <= i < len(HELP_TOPICS):
            self.view.setHtml(HELP_TOPICS[i][2])

    def show_topic(self, key, anchor=None):
        if key in self.keys:
            i = self.keys.index(key)
            if self.list.currentRow() != i:
                self.list.setCurrentRow(i)
            if anchor:
                self.view.scrollToAnchor(anchor)
        self.show(); self.raise_(); self.activateWindow()


# --------------------------------------------------------------------------- satellite photo underlay
BASEMAP_URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg"
BASEMAP_MAXZ = 14
BASEMAP_ATTRIB = ("Satellite photo: <a href='https://s2maps.eu'>Sentinel-2 cloudless - s2maps.eu</a> by "
                  "<a href='https://eox.at'>EOX IT Services GmbH</a> (contains modified Copernicus Sentinel data 2024)")
USER_AGENT = "DancingWithTheSARs/1.0 (NASA Space Apps 2026 open-source viewer; github.com/Meritas-V)"


def fetch_tile(z, x, y, cache_dir):
    p = Path(cache_dir) / str(z) / str(y) / f"{x}.jpg"
    if p.exists():
        return p.read_bytes()
    req = urllib.request.Request(BASEMAP_URL.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = r.read()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return data


def mercator_xy(lon, lat, z):
    n = 2.0 ** z
    s = np.sin(np.radians(np.clip(lat, -85.05, 85.05)))
    return (lon + 180.0) / 360.0 * n, (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n


def build_basemap(res, cache_dir, progress=None, region=None, step=1, max_tiles=160):
    """Sample web-mercator tiles onto the radar picture's pixel grid. Returns uint8 (rows, cols, 3).

    region = (row0, row1, col0, col1) in full-resolution pixels (default: everything), step = take every
    step-th pixel (used for the reduced overview). Only that grid is built, so memory stays small."""
    if region is None:
        region = (0, res.nr, 0, res.nc)
    r0_, r1_, c0_, c1_ = region
    rows = r0_ + (np.arange((r1_ - r0_) // step) + 0.5) * step
    cols = c0_ + (np.arange((c1_ - c0_) // step) + 0.5) * step
    X = res.x_min + cols * res.pix
    Y = res.y_max - rows * res.pix
    XX, YY = np.meshgrid(X, Y)
    lon, lat = res.to_ll.transform(XX, YY)
    del XX, YY
    lat_c = float(np.nanmean(lat))
    z = int(np.ceil(np.log2(156543.03392 * np.cos(np.radians(lat_c)) / (res.pix * step))))
    z = min(max(z, 3), BASEMAP_MAXZ)
    while True:
        tx, ty = mercator_xy(lon, lat, z)
        tx0, tx1, ty0, ty1 = int(np.floor(tx.min())), int(np.floor(tx.max())), int(np.floor(ty.min())), int(np.floor(ty.max()))
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= max_tiles or z <= 3:
            break
        z -= 1
    del lon, lat
    coords = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    mosaic = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), 40, np.uint8)
    done = [0]; failed = [0]

    def one(c):
        x, y = c
        try:
            im = Image.open(io.BytesIO(fetch_tile(z, x, y, cache_dir))).convert("RGB")
            return c, np.asarray(im.resize((256, 256)))
        except Exception:
            return c, None

    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for (x, y), arr in ex.map(one, coords):
            done[0] += 1
            if arr is None:
                failed[0] += 1
            else:
                mosaic[(y - ty0) * 256:(y - ty0 + 1) * 256, (x - tx0) * 256:(x - tx0 + 1) * 256] = arr
            if progress:
                progress(done[0], len(coords))
    if failed[0] > len(coords) // 2:
        raise RuntimeError(f"{failed[0]} of {len(coords)} tiles could not be downloaded (offline, or the server refused).")
    H, W = mosaic.shape[:2]
    px = (tx - tx0) * 256 - 0.5; py = (ty - ty0) * 256 - 0.5
    out = np.empty((px.shape[0], px.shape[1], 3), np.uint8)
    chunk = 200
    for r0 in range(0, px.shape[0], chunk):
        sx, sy = px[r0:r0 + chunk], py[r0:r0 + chunk]
        x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
        fx = (sx - x0).astype(np.float32)[..., None]; fy = (sy - y0).astype(np.float32)[..., None]
        x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
        y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
        v = (mosaic[y0c, x0c] * (1 - fx) * (1 - fy) + mosaic[y0c, x1c] * fx * (1 - fy) +
             mosaic[y1c, x0c] * (1 - fx) * fy + mosaic[y1c, x1c] * fx * fy)
        out[r0:r0 + chunk] = np.clip(v, 0, 255).astype(np.uint8)
    return out


class BasemapWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    ready = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, res, cache_dir, region=None, step=1, max_tiles=160):
        super().__init__()
        self.res, self.cache_dir, self.region, self.step, self.max_tiles = res, cache_dir, region, step, max_tiles
        self.stop = False

    def _progress(self, a, b):
        if self.stop:
            raise RuntimeError("cancelled")
        self.progress.emit(a, b)

    def run(self):
        try:
            self.ready.emit(build_basemap(self.res, self.cache_dir, self._progress, self.region, self.step, self.max_tiles))
        except Exception as e:
            self.failed.emit(str(e))


# --------------------------------------------------------------------------- main window
class SliderBar(QtWidgets.QWidget):
    """Slider + date labels placed by explicit geometry under the picture.

    Nothing in here has a layout, so changing where the slider sits can never change the size of
    the picture above it (that feedback loop made the first version jump around).
    """

    H_FULL = 84

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(0)
        self.setFixedHeight(self.H_FULL)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, 1000); self.slider.setValue(500)
        self.slider.setToolTip("Drag to wipe between BEFORE (left) and AFTER (right)")
        self.slider.setStyleSheet("QSlider { background: transparent; border: none; }")
        self.lb, self.lm, self.la = QtWidgets.QLabel(self), QtWidgets.QLabel(self), QtWidgets.QLabel(self)
        for lb in (self.lb, self.lm, self.la):
            lb.setTextFormat(QtCore.Qt.RichText)
        self.lb.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.lm.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.la.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.span = (0, 0)
        self.set_parts(True, True)

    def set_parts(self, slider, dates):
        self.slider.setVisible(slider)
        for lb in (self.lb, self.lm, self.la):
            lb.setVisible(dates)
        self.setFixedHeight(self.H_FULL if (slider or dates) else 2)

    def set_span(self, x0, x1):
        span = (int(x0), int(x1))
        if span != self.span:
            self.span = span
            self.place()

    def place(self):
        x0, x1 = self.span
        w = max(0, x1 - x0)
        h = HANDLE // 2
        self.slider.setGeometry(x0 - h, 6, w + 2 * h, 32)
        third = max(1, w // 3)
        self.lb.setGeometry(x0, 44, third, 40)
        self.lm.setGeometry(x0 + third, 44, third, 40)
        self.la.setGeometry(x0 + 2 * third, 44, w - 2 * third, 40)


class MainWindow(QtWidgets.QMainWindow):
    detail_ready = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dancing with the SARs - NISAR change viewer")
        scr = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1480, scr.width() - 60), min(900, scr.height() - 80))
        self.res = None
        self.cache = {}
        self._sync = False
        self.help_win = None
        self.base_cache = {}
        self.base_worker = None
        self.cache_dir = Path.cwd() / "cache" / "tiles" / "eox_s2cloudless_2024"

        # ---- left: picture, slider (same width as the picture), dates, attribution
        self.view = ImageView()
        self.view.hovered.connect(self.on_hover)
        self.view.vb.sigRangeChanged.connect(self.align_slider)
        self.view.vb.sigRangeChanged.connect(self.sync_slider)
        self.view.resized.connect(self.align_slider)
        self.base_item = pg.ImageItem()
        self.base_item.setZValue(-10)
        self.base_item.setVisible(False)
        self.view.vb.addItem(self.base_item)
        self.base_item_d = pg.ImageItem()               # satellite photo for the zoomed-in crop
        self.base_item_d.setZValue(-9); self.base_item_d.setVisible(False)
        self.view.vb.addItem(self.base_item_d)
        self.dcache = None                              # full-resolution crop currently drawn
        self.dbase_busy = False
        self.dbase_worker = None
        self._rgba_cache = {}
        self._dgen = 0
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.detail_ready.connect(self._detail_done)
        self.dtimer = QtCore.QTimer(self); self.dtimer.setSingleShot(True); self.dtimer.setInterval(60)
        self.dtimer.timeout.connect(self.update_detail)
        self.view.vb.sigRangeChanged.connect(lambda *a: self.dtimer.start())

        self.slider_bar = SliderBar()
        self.slider_bar.setAttribute(QtCore.Qt.WA_StyledBackground, False)
        self.slider = self.slider_bar.slider
        self.slider.valueChanged.connect(self.on_slider)
        self.slider.sliderReleased.connect(self.align_slider)
        self.sl_b, self.sl_mid, self.sl_a = self.slider_bar.lb, self.slider_bar.lm, self.slider_bar.la
        self._align_timer = QtCore.QTimer(self); self._align_timer.setSingleShot(True); self._align_timer.setInterval(0)
        self._align_timer.timeout.connect(self._align_now)
        self.attrib = QtWidgets.QLabel(BASEMAP_ATTRIB)
        self.attrib.setOpenExternalLinks(True); self.attrib.setTextFormat(QtCore.Qt.RichText)
        self.attrib.setStyleSheet("color:#8d93a0; font-size:11px; padding-left:8px"); self.attrib.setVisible(False)
        self.attrib.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        bottom = QtWidgets.QWidget(); bottom.setObjectName('bottombar'); bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 4, 0, 4); bl.setSpacing(0)
        bl.addWidget(self.slider_bar); bl.addWidget(self.attrib)
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(0)
        ll.addWidget(self.view, 1); ll.addWidget(bottom)

        # ---- right: controls, hover card, tabs
        side = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(side); sl.setContentsMargins(10, 10, 10, 6); sl.setSpacing(8)
        self.layer = QtWidgets.QComboBox(); self.layer.addItems(LAYERS)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["Swipe: before | after (slider below the picture)", "Before only", "After only", "Swipe: satellite photo | radar (after)"])
        self.clip = QtWidgets.QDoubleSpinBox(); self.clip.setRange(1, 20); self.clip.setValue(6.0); self.clip.setSuffix(" dB")
        self.clip.setToolTip("Change maps saturate at +/- this many decibels")
        self.fit_btn = QtWidgets.QPushButton("Fit picture to window"); self.fit_btn.clicked.connect(self.fit)
        self.base_chk = QtWidgets.QCheckBox("Show satellite photo underneath")
        self.base_chk.setToolTip("Downloads Sentinel-2 cloudless tiles (needs internet, first time only)")
        self.base_chk.toggled.connect(self.on_base_toggle)
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.opacity.setRange(10, 100); self.opacity.setValue(100)
        self.opacity.setStyleSheet("QSlider { background: transparent; border: none; }"); self.opacity.setEnabled(False); self.opacity.valueChanged.connect(self.apply_opacity)
        self.opacity_lbl = QtWidgets.QLabel("Radar opacity: 100%")
        self.opacity.setVisible(False); self.opacity_lbl.setVisible(False)
        self.base_status = QtWidgets.QLabel(""); self.base_status.setStyleSheet("color:#a9afbd; font-size:11px")
        self.base_status.setWordWrap(True)
        c1 = QtWidgets.QFrame(); c1.setObjectName("card"); cl = QtWidgets.QVBoxLayout(c1)
        cl.addWidget(heading("Layer")); cl.addWidget(self.layer)
        cl.addWidget(heading("Compare")); cl.addWidget(self.mode)
        cl.addWidget(heading("Change colour range (±)")); cl.addWidget(self.clip)
        self.bar = QtWidgets.QLabel(); self.bar.setFixedHeight(16); self.bar.setScaledContents(True)
        self.bar_txt = QtWidgets.QLabel(); self.bar_txt.setStyleSheet("color:#a9afbd; font-size:11px")
        cl.addWidget(self.bar); cl.addWidget(self.bar_txt)
        cl.addWidget(heading("Satellite photo")); cl.addWidget(self.base_chk)
        cl.addWidget(self.opacity_lbl); cl.addWidget(self.opacity); cl.addWidget(self.base_status)
        cl.addWidget(self.fit_btn)
        gb = QtWidgets.QToolButton(); gb.setObjectName("helpbig"); gb.setIcon(help_icon(64)); gb.setIconSize(QtCore.QSize(34, 34))
        gb.setToolTip("Guide, explanations and sources (F1)"); gb.setCursor(QtCore.Qt.PointingHandCursor)
        gb.clicked.connect(lambda: self.open_help()); gb.hide()
        title = QtWidgets.QLabel("<b>Dancing with the SARs</b><br><span style='color:#8d93a0; font-size:11px'>NISAR radar change viewer</span>")
        title.setTextFormat(QtCore.Qt.RichText)
        head = QtWidgets.QHBoxLayout(); head.setContentsMargins(4, 0, 0, 0); head.addWidget(title, 1); head.addWidget(gb)
        sl.insertLayout(0, head)
        sl.addWidget(c1)

        c2 = QtWidgets.QFrame(); c2.setObjectName("card"); hl = QtWidgets.QVBoxLayout(c2)
        hl.addWidget(heading("Under the cursor"))
        self.hover = QtWidgets.QLabel("Move the mouse over the picture."); self.hover.setTextFormat(QtCore.Qt.RichText)
        self.hover.setWordWrap(True); self.hover.setFixedHeight(132); self.hover.setAlignment(QtCore.Qt.AlignTop)
        hl.addWidget(self.hover)
        sl.addWidget(c2)

        self.tabs = QtWidgets.QTabWidget()
        self.viewcards, self.stats, self.weather = LegendView(), StatsPanel(), WeatherPanel()
        self.viewcards.helpRequested.connect(self.open_help)
        for cl_ in (self.stats, self.weather.cards):
            cl_.help_cb = self.open_help
        self.tabs.addTab(self.viewcards, "Legend"); self.tabs.addTab(self.stats, "Statistics")
        self.tabs.addTab(self.weather, "Weather")
        hb = QtWidgets.QToolButton(); hb.setObjectName("help"); hb.setIcon(help_icon(64)); hb.setIconSize(QtCore.QSize(22, 22))
        hb.setToolTip("Guide and sources (F1)"); hb.setCursor(QtCore.Qt.PointingHandCursor)
        hb.clicked.connect(lambda: self.open_help())
        self.tabs.setCornerWidget(hb, QtCore.Qt.TopRightCorner)
        sl.addWidget(self.tabs, 1)
        side.setMinimumWidth(440)

        split = QtWidgets.QSplitter(); split.addWidget(left); split.addWidget(side)
        split.setStretchFactor(0, 1); split.setSizes([1040, 440])
        self.setCentralWidget(split)
        self.readout = QtWidgets.QLabel("Open a results folder (Open results) to begin.")
        self.statusBar().addWidget(self.readout, 1)
        self.lod_lbl = QtWidgets.QLabel(""); self.lod_lbl.setStyleSheet("color:#8fb4ff; padding-right:8px")
        self.statusBar().addPermanentWidget(self.lod_lbl)

        self.ribbon = QtWidgets.QToolBar("Tools"); self.ribbon.setObjectName("ribbon")
        self.ribbon.setMovable(False); self.ribbon.setFloatable(False)
        self.ribbon.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon); self.ribbon.setIconSize(QtCore.QSize(26, 26))
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.ribbon)

        def tool(kind, text, tip, slot, key=None, checkable=False):
            act = QtGui.QAction(ribbon_icon(kind), text, self); act.setToolTip(f"{tip}" + (f" ({key})" if key else ""))
            act.setCheckable(checkable)
            if key:
                act.setShortcut(key)
            act.triggered.connect(slot)
            self.ribbon.addAction(act)
            return act
        tool("open", "Open results", "Open a results folder", self.open_dialog, "Ctrl+O")
        tool("process", "Process pair", "Process two NISAR GCOV files into a results folder", self.process_dialog, "Ctrl+N")
        tool("save", "Save picture", "Save the current view as a PNG", self.save_png, "Ctrl+S")
        self.ribbon.addSeparator()
        tool("fit", "Fit", "Fit the whole picture in the window", self.fit)
        self.frame_act = tool("frame", "Frame", "Show or hide the outline of the picture", self.toggle_frame, checkable=True)
        self.frame_act.setChecked(True)
        self.ribbon.addSeparator()
        tool("terms", "Terms", "Short glossary of the terms used", lambda: self.open_help("terms"))
        tool("help", "Guide", "Guide, explanations and sources", lambda: self.open_help(), "F1")
        self.layer.currentIndexChanged.connect(self.rebuild)
        self.mode.currentIndexChanged.connect(self.on_mode)
        self.clip.valueChanged.connect(self.rebuild)

    # ---- help window
    def open_help(self, topic=None):
        if self.help_win is None:
            self.help_win = HelpWindow(self)
        if topic == "reading":
            self.help_win.show_topic("reading", f"layer{self.layer.currentIndex()}")
        else:
            self.help_win.show_topic(topic or "start")

    # ---- loading
    def open_dialog(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Open a results folder (contains info.json)",
                                                       str(Path.cwd() / "out"))
        if p:
            self.load(p)

    def process_dialog(self):
        dlg = ProcessDialog(self)
        dlg.finished_ok.connect(lambda p: (self.load(p), dlg.accept()))
        dlg.exec()

    def save_png(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save view", "sar_view.png", "PNG (*.png)")
        if p:
            self.view.grab().save(p)

    def toggle_frame(self, on):
        self.view.frame.setVisible(bool(on))

    def fit(self):
        if self.res:
            self.view.vb.setRange(xRange=(0, self.res.nc), yRange=(0, self.res.nr), padding=0.02)

    def load(self, folder):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.res = None; self.cache = {}; self.dcache = None
            self.res = Results(folder, progress=lambda k: (self.readout.setText(f"Reading {k} ..."), QtWidgets.QApplication.processEvents()))
        except MemoryError:
            QtWidgets.QMessageBox.warning(self, "Not enough memory", f"{folder}\n\nThis result is too big for the free memory. "
                                          "Process it again with a larger block size (Process pair).")
            return
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Cannot open", f"{folder}\n\n{e}\n\nPick a folder made by process_pair.py "
                                          "(it contains info.json and stack.npz).")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        r, i = self.res, self.res.info
        self.setWindowTitle(f"Dancing with the SARs - {Path(folder).name}")
        gap = days_between(i["before"]["start"], i["after"]["start"])
        self.gap_days = gap
        self.set_labels()
        self.weather.set_results(r)
        self.stats.set_results(r)
        self.view.line.setBounds((0, r.nc))
        self._sync = True; self.view.line.setValue(r.nc / 2); self.slider.setValue(500); self._sync = False
        self.base_item.setVisible(False); self.base_item_d.setVisible(False)
        self.readout.setText("Ready. Hover over the picture.")
        self.view.set_frame(self.res.nc, self.res.nr)
        self.rebuild()
        self.fit()
        self.align_slider()
        QtCore.QTimer.singleShot(60, self.align_slider)
        if self.base_chk.isChecked():
            self.start_basemap()

    def set_labels(self):
        if not self.res:
            return
        i = self.res.info
        if self.mode.currentIndex() == 3:
            left, mid = ("SATELLITE PHOTO", "Sentinel-2 cloudless 2024"), "optical photo vs radar"
        else:
            left, mid = ("BEFORE", nice_date(i["before"]["start"])), f"{self.gap_days} days apart"
        self.sl_b.setText(f"<span style='color:#7fa6ff; font-size:11px'>{left[0]}</span><br>"
                          f"<span style='color:#7fa6ff; font-size:16px'><b>{left[1]}</b></span>")
        self.sl_a.setText(f"<span style='color:#ffb84a; font-size:11px'>AFTER (RADAR)</span><br>"
                          f"<span style='color:#ffb84a; font-size:16px'><b>{nice_date(i['after']['start'])}</b></span>")
        self.sl_mid.setText(f"<span style='color:#a9afbd'>{mid}</span>")

    # ---- slider (the only way to move the line)
    def visible_x(self):
        """Left and right image column currently on screen (the slider covers exactly this part)."""
        (x0, x1), _ = self.view.vb.viewRange()
        lo, hi = max(0.0, x0), min(float(self.res.nc), x1)
        return (lo, hi) if hi - lo > 1 else (0.0, float(self.res.nc))

    def on_slider(self, v):
        if self._sync or not self.res:
            return
        lo, hi = self.visible_x()
        self.view.line.setValue(lo + v / 1000 * (hi - lo))
        self.recompose()

    def sync_slider(self, *_):
        """After zooming or panning the line stays where it is on the picture; the slider follows it."""
        if not self.res or self.slider.isSliderDown():
            return
        lo, hi = self.visible_x()
        v = int(round(np.clip((self.view.line.value() - lo) / (hi - lo), 0, 1) * 1000))
        if v != self.slider.value():
            self._sync = True; self.slider.setValue(v); self._sync = False

    def align_slider(self, *_):
        """Ask for the slider to be re-aligned (coalesced: many triggers, one update)."""
        self._align_timer.start()

    def _align_now(self):
        """Make the slider and the dates span exactly the picture's left and right edges."""
        if not self.res or self.slider.isSliderDown():
            return
        vb = self.view.vb
        p0 = self.view.mapFromScene(vb.mapViewToScene(QtCore.QPointF(0, 0)))
        p1 = self.view.mapFromScene(vb.mapViewToScene(QtCore.QPointF(self.res.nc, self.res.nr)))
        W = self.view.width()
        x0, x1 = max(0, min(p0.x(), p1.x())), min(W, max(p0.x(), p1.x()))
        if x1 - x0 > 20:
            self.slider_bar.set_span(x0, x1)

    # ---- satellite photo underlay
    def on_base_toggle(self, on):
        self.opacity.setEnabled(on)
        self.opacity.setVisible(on); self.opacity_lbl.setVisible(on)
        if on:
            if self.opacity.value() == 100:
                self.opacity.setValue(60)
            self.start_basemap()
        else:
            self.base_item.setVisible(False); self.base_item_d.setVisible(False)
            self.attrib.setVisible(self.mode.currentIndex() == 3)
            self.base_status.setText("")
            self.apply_opacity()

    def key(self):
        return str(self.res.folder.resolve()) if self.res else None

    def want_base(self):
        return self.base_chk.isChecked() or self.mode.currentIndex() == 3

    def start_basemap(self):
        """Satellite photo for the overview picture (the whole scene, at the overview's resolution)."""
        if not self.res:
            return
        key = self.key()
        if key in self.base_cache:
            self.show_basemap(self.base_cache[key]); self.request_detail_base(); return
        r = self.res
        self.base_status.setText("Loading satellite photo...")
        w = BasemapWorker(r, self.cache_dir, region=(0, r.nr_o * r.f, 0, r.nc_o * r.f), step=r.f)
        w.progress.connect(lambda a, b: self.base_status.setText(f"Loading satellite photo: {a} of {b} tiles..."))
        w.ready.connect(lambda arr, k=key: self._base_ready(k, arr))
        w.failed.connect(lambda msg: self.base_status.setText(
            "Could not load the satellite photo: " + msg + " The viewer works without it.") if msg != "cancelled" else None)
        self.base_worker = w
        w.start()

    def _base_ready(self, key, arr):
        self.base_cache[key] = arr
        if self.res and self.key() == key:
            if self.base_chk.isChecked():
                self.show_basemap(arr)
            if self.mode.currentIndex() == 3:
                self.base_status.setText("Satellite photo ready.")
                self.recompose()
            self.request_detail_base()

    def show_basemap(self, arr):
        r = self.res
        self._put(self.base_item, arr, 0, 0, r.nc_o * r.f, r.nr_o * r.f)
        self.base_item.setVisible(True)
        self.attrib.setVisible(True)
        self.base_status.setText("Satellite photo ready. Transparent parts of the radar picture have no radar data.")
        self.apply_opacity()

    def apply_opacity(self, *_):
        v = self.opacity.value()
        self.opacity_lbl.setText(f"Radar opacity: {v}%")
        op = v / 100 if self.base_chk.isChecked() else 1.0
        self.view.img.setOpacity(op); self.view.img_d.setOpacity(op)
        # with a see-through radar layer, drawing overview and crop on top of each other would double the radar
        self.view.img.setVisible(not (self.dcache and op < 1.0))

    # ---- deep zoom: full-resolution crop of what is on screen
    def _put(self, item, arr, x0, y0, w, h):
        item.setImage(arr, autoLevels=False, levels=(0, 255))
        item.setRect(QtCore.QRectF(x0, y0, w, h))

    def _rgba_of(self, rgb):
        """Satellite RGB -> RGBA (cached for the overview photo, which is composed again on every slider move)."""
        if rgb is None:
            return None
        hit = self._rgba_cache.get(id(rgb))
        if hit is not None and hit[0] is rgb:
            return hit[1]
        out = np.empty(rgb.shape[:2] + (4,), np.uint8); out[..., :3] = rgb; out[..., 3] = 255
        if len(self._rgba_cache) >= 3:
            self._rgba_cache.pop(next(iter(self._rgba_cache)))
        self._rgba_cache[id(rgb)] = (rgb, out)
        return out

    def update_detail(self):
        """Redraw the visible part at full resolution when it is small enough (called after zooming/panning)."""
        r = self.res
        self._dgen += 1                                  # any job still running is now out of date
        if not r or r.f == 1 or not self.cache:
            self.dcache = None
            self.recompose_detail(); self.set_lod_label(); return
        (x0, x1), (y0, y1) = self.view.vb.viewRange()
        c_lo, c_hi = max(0, int(np.floor(x0))), min(r.nc, int(np.ceil(x1)))
        r_lo, r_hi = max(0, int(np.floor(y0))), min(r.nr, int(np.ceil(y1)))
        L, clip = self.layer.currentIndex(), self.clip.value()
        if c_hi - c_lo < 2 or r_hi - r_lo < 2 or (c_hi - c_lo) * (r_hi - r_lo) > DETAIL_MAX:
            self.dcache = None
            self.recompose_detail(); self.set_lod_label(); return
        d = self.dcache
        if d and d["L"] == L and d["clip"] == clip:
            a0, a1, b0, b1 = d["rect"]
            if a0 <= r_lo and a1 >= r_hi and b0 <= c_lo and b1 >= c_hi:
                self.set_lod_label(); return           # the crop already covers the screen
        mh, mw = int((r_hi - r_lo) * 0.5), int((c_hi - c_lo) * 0.5)
        rect = (max(0, r_lo - mh), min(r.nr, r_hi + mh), max(0, c_lo - mw), min(r.nc, c_hi + mw))
        if (rect[1] - rect[0]) * (rect[3] - rect[2]) > DETAIL_MAX:
            rect = (r_lo, r_hi, c_lo, c_hi)
        self._pool.submit(self._detail_job, self._dgen, r, L, clip, rect)   # computed off the GUI thread

    def _detail_job(self, gen, r, L, clip, rect):
        """Runs in a worker thread: builds the full-resolution picture for `rect` without blocking the window."""
        if gen != self._dgen:
            return
        try:
            after = r.layer_rgba(L, "after", clip, rect)
            if gen != self._dgen:
                return
            before = after if L in (3, 4) else r.layer_rgba(L, "before", clip, rect)
            if gen == self._dgen:
                self.detail_ready.emit((gen, r, L, clip, rect, before, after))
        except Exception:
            pass

    def _detail_done(self, pkg):
        gen, r, L, clip, rect, before, after = pkg
        if gen != self._dgen or r is not self.res or L != self.layer.currentIndex() or clip != self.clip.value():
            return
        self.dcache = {"rect": rect, "L": L, "clip": clip, "before": before, "after": after, "base": None}
        self.recompose_detail(); self.set_lod_label()
        self.request_detail_base()

    def set_lod_label(self):
        r = self.res
        if not r or r.f == 1:
            self.lod_lbl.setText(""); return
        self.lod_lbl.setText(f"Full detail ({r.pix:.0f} m pixels)" if self.dcache else
                             f"Overview ({r.f}x{r.f} averaged). Zoom in for every {r.pix:.0f} m pixel")

    def request_detail_base(self):
        d = self.dcache
        if not d or not self.want_base() or d.get("base") is not None or self.dbase_busy or not self.res:
            return
        rect = d["rect"]
        self.dbase_busy = True
        w = BasemapWorker(self.res, self.cache_dir, region=rect, step=1, max_tiles=220)
        w.ready.connect(lambda arr, rect=rect: self._dbase_ready(rect, arr))
        w.failed.connect(self._dbase_failed)
        self.dbase_worker = w
        w.start()

    def _dbase_ready(self, rect, arr):
        self.dbase_busy = False
        d = self.dcache
        if d and d["rect"] == rect:
            d["base"] = arr
            self.recompose_detail()
        else:
            self.request_detail_base()

    def _dbase_failed(self, msg):
        self.dbase_busy = False
        if msg != "cancelled":
            self.base_status.setText("Sharper satellite photo for this zoom could not be loaded: " + msg)

    def closeEvent(self, e):
        self._dgen += 1
        self._pool.shutdown(wait=False, cancel_futures=True)
        for w in (self.base_worker, self.dbase_worker):
            if w is not None and w.isRunning():
                w.stop = True
                w.wait(8000)
        super().closeEvent(e)

    # ---- drawing
    def update_colourbar(self, change):
        self.bar.setVisible(change); self.bar_txt.setVisible(change)
        if change:
            c = self.clip.value()
            ramp = np.ascontiguousarray(pp.change_rgb(np.tile(np.linspace(-c, c, 256), (1, 1)), c))
            img = QtGui.QImage(ramp.tobytes(), 256, 1, 256 * 3, QtGui.QImage.Format_RGB888).copy()
            self.bar.setPixmap(QtGui.QPixmap.fromImage(img))
            self.bar_txt.setText(f"−{c:g} dB darker            no change            brighter +{c:g} dB")

    def refresh_info(self):
        if not self.res:
            return
        self.viewcards.set_cards(legend_cards(self.res, self.layer.currentIndex(), self.mode.currentIndex(),
                                              self.clip.value(), self.base_chk.isChecked()))

    def on_mode(self, *_):
        self.set_labels()
        if self.mode.currentIndex() == 3 and self.res:
            if self.key() not in self.base_cache:
                self.start_basemap()
            else:
                self.request_detail_base()
            self.attrib.setVisible(True)
        elif not self.base_chk.isChecked():
            self.attrib.setVisible(False)
        self.recompose()
        self.refresh_info()

    def rebuild(self, *_):
        L = self.layer.currentIndex()
        change = L in (3, 4)
        self.update_colourbar(change)
        if not self.res:
            return
        clip = self.clip.value()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.cache = {"before": self.res.layer_rgba(L, "before", clip), "after": self.res.layer_rgba(L, "after", clip)}
            self.dcache = None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.mode.setEnabled(not change)
        # no comparison on the change maps, so no slider and no dates under the picture
        self.slider_bar.set_parts(not change, not change)
        self.recompose()
        self.refresh_info()
        self.update_detail()

    def _compose(self, c, split, base):
        """Picture for the current compare mode. c = {'before','after'} RGBA, split = column of the swipe line."""
        L = self.layer.currentIndex()
        change = L in (3, 4)
        m = self.mode.currentIndex()
        if change or m == 2:
            return c["after"]
        if m == 1:
            return c["before"]
        img = c["after"].copy()
        if m == 3:
            img[:, :split] = base[:, :split] if base is not None else 0
        else:
            img[:, :split] = c["before"][:, :split]
        return img

    def recompose(self, *_):
        r = self.res
        if not r or not self.cache:
            return
        L = self.layer.currentIndex()
        change = L in (3, 4)
        m = self.mode.currentIndex()
        swipe = (not change) and m in (0, 3)
        self.view.line.setVisible(swipe)
        self.slider.setEnabled(swipe)
        split = int(np.clip(self.view.line.value() / r.f, 0, r.nc_o))
        base = self._rgba_of(self.base_cache.get(self.key())) if m == 3 else None
        self._put(self.view.img, self._compose(self.cache, split, base), 0, 0, r.nc_o * r.f, r.nr_o * r.f)
        self.recompose_detail()

    def recompose_detail(self):
        d = self.dcache
        item = self.view.img_d
        if not d or not self.res:
            item.setVisible(False); self.base_item_d.setVisible(False)
            self.base_item.setVisible(bool(self.res) and self.base_chk.isChecked() and self.key() in self.base_cache)
            self.apply_opacity()
            return
        r0, r1, c0, c1 = d["rect"]
        w, h = c1 - c0, r1 - r0
        split = int(np.clip(self.view.line.value() - c0, 0, w))
        base = d["base"]
        img = self._compose(d, split, self._rgba_of(base) if self.mode.currentIndex() == 3 else None)
        self._put(item, img, c0, r0, w, h)
        item.setVisible(True)
        show_base = base is not None and self.base_chk.isChecked()
        if show_base:
            self._put(self.base_item_d, base, c0, r0, w, h)
        self.base_item_d.setVisible(show_base)
        self.base_item.setVisible(self.base_chk.isChecked() and self.key() in self.base_cache and not show_base)
        self.apply_opacity()

    def on_hover(self, col, row):
        r = self.res
        if not r or not (0 <= row < r.nr and 0 <= col < r.nc):
            return
        lon, lat = r.lonlat(row, col)
        head = f"<b>{lat:.4f}° N, {lon:.4f}° E</b>"
        if not r.valid[row, col]:
            self.hover.setText(head + "<br><span style='color:#8d93a0'>Outside the scene / no valid data</span>")
            self.readout.setText(f"lat {lat:.4f}  lon {lon:.4f}  |  no valid data")
            return
        A = r.a
        lines, short = [head], []
        for pol, nm, desc in (("hv", CROSS, "vegetation"), ("hh", LIKE, "surface")):
            b, a = float(A[f"{pol}_before"][row, col]), float(A[f"{pol}_after"][row, col])
            loc = float(A[f"change_{pol}"][row, col])
            colour = "#7fa6ff" if loc < -1 else "#ff8a7a" if loc > 1 else "#cfd3dc"
            word = verdict(loc)
            tail = "as the scene average" if word == "about the same" else "than the scene average"
            lines.append(f"{nm} ({desc}): {b:.1f} → {a:.1f} dB<br>"
                         f"&nbsp;&nbsp;<span style='color:{colour}'><b>{loc:+.1f} dB, {word}</b></span> {tail}")
            short.append(f"{nm.lower()} {b:.1f} to {a:.1f} dB (local {loc:+.1f})")
        self.hover.setText("<br>".join(lines))
        self.readout.setText(f"lat {lat:.4f}  lon {lon:.4f}  |  " + "  |  ".join(short))


def apply_dark(app):
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    c = QtGui.QColor
    p.setColor(QtGui.QPalette.Window, c("#202124")); p.setColor(QtGui.QPalette.WindowText, c("#e8eaed"))
    p.setColor(QtGui.QPalette.Base, c("#17181b")); p.setColor(QtGui.QPalette.AlternateBase, c("#202124"))
    p.setColor(QtGui.QPalette.Text, c("#e8eaed")); p.setColor(QtGui.QPalette.Button, c("#2b2d31"))
    p.setColor(QtGui.QPalette.ButtonText, c("#e8eaed")); p.setColor(QtGui.QPalette.Highlight, c("#3b6fe0"))
    p.setColor(QtGui.QPalette.HighlightedText, c("#ffffff")); p.setColor(QtGui.QPalette.ToolTipBase, c("#2b2d31"))
    p.setColor(QtGui.QPalette.ToolTipText, c("#e8eaed"))
    p.setColor(QtGui.QPalette.Link, c("#8fb4ff")); p.setColor(QtGui.QPalette.LinkVisited, c("#b9a5ff"))
    app.setPalette(p)
    app.setStyleSheet(STYLE)


def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark(app)
    w = MainWindow()
    w.show()
    start = sys.argv[1] if len(sys.argv) > 1 else None
    if not start:
        for cand in ("out/overview", "demo/overview", "out/detail3", "out/detail"):
            if (Path(cand) / "info.json").exists():
                start = cand
                break
    if start:
        w.load(start)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
