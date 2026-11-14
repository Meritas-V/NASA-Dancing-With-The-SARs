"""
process_pair.py - turn TWO NISAR GCOV files (same track + frame) into change-detection images.

Usage (PowerShell, venv active):
  python process_pair.py BEFORE.h5 AFTER.h5 --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX --block 4 --name overview

What it does, step by step (each step is a labelled block below):
  1. SAFETY CHECK   - both files must be the same track, frame and pass direction, on the same grid.
                      If not, the pixels do not line up and any "change" would be nonsense -> we stop.
  2. WINDOW         - convert your lon/lat box to UTM metres, then to pixel rows/cols. Only that window is read
                      from disk (a full layer is 1.3 GB; the window is a few MB to ~150 MB).
  3. MASK           - keep only pixels the file marks as valid (mask 1..5 = sub-swath number of fully focused data). 0 = invalid or partially focused, 255 = outside footprint (spec JPL D-102274 Rev E, sec. 4.3.3).
  4. BLOCK AVERAGE  - average block x block pixels in LINEAR power (never average dB!), valid pixels only.
  5. dB             - 10*log10(power). Radar brightness spans orders of magnitude, so we use a log scale.
  6. CHANGE         - after_dB - before_dB. Then subtract the median shift so a regional brightness offset
                      (wet/frozen ground, calibration) doesn't tint the whole map.
  7. OUTPUT         - PNGs (transparent where invalid), stack.npz (arrays for the future app), info.json.
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image
from pyproj import Transformer

GRID = "/science/LSAR/GCOV/grids/frequencyA"
IDENT = "/science/LSAR/identification"
DB_FORMULA = "/science/LSAR/GCOV/metadata/ceosAnalysisReadyData/outputBackscatterDecibelConversionFormula"


# ----------------------------------------------------------------------------- helpers
def dec(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return dec(v.item())
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_ident(f):
    g = f[IDENT]
    return {
        "track": int(dec(g["trackNumber"][()])),
        "frame": int(dec(g["frameNumber"][()])),
        "pass": str(dec(g["orbitPassDirection"][()])),
        "start": str(dec(g["zeroDopplerStartTime"][()])),
    }


def check_gcov(f, label):
    """Return a list of plain-language problems that stop this file from being used (empty list = fine)."""
    out = []
    kind = ""
    try:
        for k in ("productType", "productSpecificationVersion"):
            pass
        kind = str(dec(f[IDENT]["productType"][()]))
    except Exception:
        pass
    if GRID not in f:
        what = f"it is a {kind} product" if kind and kind != "GCOV" else "it has no frequency A grid (an S-band, frequency-B-only or non-GCOV file)"
        out.append(f"{label}: {what}. This tool reads L-band GCOV files (name contains _GCOV_) with frequency A.")
        return out
    g = f[GRID]
    for need in ("HHHH", "HVHV", "mask", "xCoordinates", "yCoordinates", "projection"):
        if need not in g:
            out.append(f"{label}: layer {need} is missing. The tool needs a dual-polarised HH/HV GCOV file.")
    return out


def read_axes(f):
    g = f[GRID]
    return g["xCoordinates"][()], g["yCoordinates"][()], int(dec(g["projection"][()]))


def block_mean(power, valid, b, min_frac=0.5):
    """Average b x b blocks in linear power using valid pixels only.
    A block is kept only if at least min_frac of its pixels were valid."""
    nr, nc = power.shape[0] // b, power.shape[1] // b
    p = np.where(valid, power, 0.0).astype(np.float64)[: nr * b, : nc * b]
    v = valid[: nr * b, : nc * b]
    s = p.reshape(nr, b, nc, b).sum(axis=(1, 3))
    n = v.reshape(nr, b, nc, b).sum(axis=(1, 3))
    ok = n >= max(1, min_frac * b * b)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
    mean[~ok] = np.nan
    return mean, ok


def to_db(power):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10.0 * np.log10(power)


def u8(a, lo, hi):
    a = np.nan_to_num((a - lo) / (hi - lo), nan=0.0)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def save_rgba(rgb, ok, path):
    alpha = (ok * 255).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path)


def save_gray(db, ok, lo, hi, path):
    g = u8(db, lo, hi)
    save_rgba(np.dstack([g, g, g]), ok, path)


def save_rgb(hh, hv, ok, r_hh, r_hv, r_ratio, path):
    r = u8(hh, *r_hh)
    g = u8(hv, *r_hv)
    b = u8(hh - hv, *r_ratio)
    save_rgba(np.dstack([r, g, b]), ok, path)


def change_rgb(ch, clip):
    """Diverging colours: blue = got darker (less backscatter), white = no change, red = got brighter."""
    t = np.clip(np.nan_to_num(ch / clip, nan=0.0), -1, 1)[..., None]
    white = np.array([255, 255, 255], float)
    blue = np.array([20, 80, 210], float)
    red = np.array([210, 40, 30], float)
    return np.where(t < 0, white * (1 + t) + blue * (-t), white * (1 - t) + red * t).astype(np.uint8)


def save_change(ch, ok, clip, path):
    save_rgba(change_rgb(ch, clip), ok, path)


def pct_range(a, ok, lo=2, hi=98):
    vals = a[ok & np.isfinite(a)]
    if vals.size == 0:
        return (-25.0, 0.0)
    p = np.percentile(vals, [lo, hi])
    return (float(p[0]), float(p[1]))


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--block", type=int, default=4, help="average block x block pixels (20 m x block). default 4 = 80 m")
    ap.add_argument("--name", default="run", help="output subfolder name")
    ap.add_argument("--out", default="out", help="output root folder")
    ap.add_argument("--clip", type=float, default=6.0, help="change map saturates at +/- this many dB")
    ap.add_argument("--no-png", action="store_true", help="skip the PNG previews (much faster on huge areas; the viewer does not need them)")
    ap.add_argument("--no-normalise", action="store_true", help="do not subtract the regional median shift")
    ap.add_argument("--force", action="store_true", help="continue even if track/frame/grid differ (results will be wrong)")
    a = ap.parse_args(argv)

    outdir = os.path.join(a.out, a.name)
    os.makedirs(outdir, exist_ok=True)

    fb = h5py.File(a.before, "r")
    fa = h5py.File(a.after, "r")

    # ---- 0. IS EACH FILE A USABLE GCOV FILE?
    bad = check_gcov(fb, "BEFORE file (" + os.path.basename(a.before) + ")") + check_gcov(fa, "AFTER file (" + os.path.basename(a.after) + ")")
    if bad:
        print("CANNOT USE THESE FILES:")
        for m in bad:
            print("  -", m)
        sys.exit(5)

    # ---- 1. SAFETY CHECK
    ib, ia = read_ident(fb), read_ident(fa)
    xb, yb, epsg_b = read_axes(fb)
    xa, ya, epsg_a = read_axes(fa)
    problems = []
    for k in ("track", "frame", "pass"):
        if ib[k] != ia[k]:
            problems.append(f"{k} differs: {ib[k]} vs {ia[k]}")
    if epsg_b != epsg_a:
        problems.append(f"projection differs: {epsg_b} vs {epsg_a}")
    if xb.shape != xa.shape or yb.shape != ya.shape or not (np.allclose(xb[[0, -1]], xa[[0, -1]]) and np.allclose(yb[[0, -1]], ya[[0, -1]])):
        problems.append("grids differ (different size or coordinates)")
    if problems:
        print("NOT COMPARABLE:")
        for p in problems:
            print("  -", p)
        if not a.force:
            print("Stopping. Pick two files from the SAME track and frame (use --force to override).")
            sys.exit(2)
    print(f"before: {ib['start']}  after: {ia['start']}  track {ib['track']} frame {ib['frame']} {ib['pass']}")
    if DB_FORMULA in fb:
        print("file's own dB formula:", dec(fb[DB_FORMULA][()]))

    # ---- 2. WINDOW (lon/lat box -> UTM metres -> pixel rows/cols)
    to_utm = Transformer.from_crs(4326, epsg_b, always_xy=True)
    to_ll = Transformer.from_crs(epsg_b, 4326, always_xy=True)
    lon0, lat0, lon1, lat1 = a.bbox
    cx, cy = to_utm.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
    dx = float(xb[1] - xb[0])          # +20
    dy = float(yb[1] - yb[0])          # -20 (row 0 is the NORTH edge)
    cols = (np.array(cx) - xb[0]) / dx
    rows = (np.array(cy) - yb[0]) / dy
    c_lo, c_hi = int(np.floor(cols.min())), int(np.ceil(cols.max())) + 1
    r_lo, r_hi = int(np.floor(rows.min())), int(np.ceil(rows.max())) + 1
    c_lo, r_lo = max(c_lo, 0), max(r_lo, 0)
    c_hi, r_hi = min(c_hi, len(xb)), min(r_hi, len(yb))
    b = a.block
    nc_out, nr_out = (c_hi - c_lo) // b, (r_hi - r_lo) // b
    if nc_out < 2 or nr_out < 2:
        print("The box is (almost) outside the file's grid. Check --bbox (lon lat order: LON_MIN LAT_MIN LON_MAX LAT_MAX).")
        sys.exit(3)
    c_hi, r_hi = c_lo + nc_out * b, r_lo + nr_out * b
    print(f"window: cols {c_lo}..{c_hi}  rows {r_lo}..{r_hi}  ->  {c_hi - c_lo} x {r_hi - r_lo} px at {abs(dx):.0f} m, "
          f"output {nc_out} x {nr_out} px at {abs(dx) * b:.0f} m")

    # ---- 3,4,5. read window, mask, block-average, dB (one date + polarisation at a time to save RAM)
    def load(f, pol):
        g = f[GRID]
        power = g[pol][r_lo:r_hi, c_lo:c_hi]
        mask = g["mask"][r_lo:r_hi, c_lo:c_hi]
        valid = (mask >= 1) & (mask <= 5) & np.isfinite(power) & (power > 0)
        mean, ok = block_mean(power, valid, b)
        return to_db(mean), ok

    hh_b, ok1 = load(fb, "HHHH")
    hv_b, ok2 = load(fb, "HVHV")
    hh_a, ok3 = load(fa, "HHHH")
    hv_a, ok4 = load(fa, "HVHV")
    ok = ok1 & ok2 & ok3 & ok4      # a pixel is usable only if valid in ALL four layers
    for arr in (hh_b, hv_b, hh_a, hv_a):
        arr[~ok] = np.nan
    print(f"valid output pixels: {ok.mean() * 100:.1f}%")
    if ok.sum() == 0:
        print("No valid pixels in this window (probably outside the scene footprint).")
        sys.exit(4)

    # ---- 6. CHANGE
    d_hh = hh_a - hh_b
    d_hv = hv_a - hv_b
    shift_hh = float(np.nanmedian(d_hh[ok]))
    shift_hv = float(np.nanmedian(d_hv[ok]))
    print(f"median shift after-before: HH {shift_hh:+.2f} dB, HV {shift_hv:+.2f} dB")
    if not a.no_normalise:
        d_hh = d_hh - shift_hh
        d_hv = d_hv - shift_hv
        print("shift removed (use --no-normalise to keep it)")

    # ---- 7. OUTPUT
    r_hh = pct_range(np.concatenate([hh_b[ok], hh_a[ok]]), np.ones(2 * int(ok.sum()), bool))
    r_hv = pct_range(np.concatenate([hv_b[ok], hv_a[ok]]), np.ones(2 * int(ok.sum()), bool))
    ratio_all = np.concatenate([(hh_b - hv_b)[ok], (hh_a - hv_a)[ok]])
    r_ratio = pct_range(ratio_all, np.ones(ratio_all.size, bool))
    p = lambda n: os.path.join(outdir, n)
    if not a.no_png:
        save_gray(hh_b, ok, *r_hh, p("hh_before.png"))
    if not a.no_png:
        save_gray(hh_a, ok, *r_hh, p("hh_after.png"))
    if not a.no_png:
        save_gray(hv_b, ok, *r_hv, p("hv_before.png"))
    if not a.no_png:
        save_gray(hv_a, ok, *r_hv, p("hv_after.png"))
    if not a.no_png:
        save_rgb(hh_b, hv_b, ok, r_hh, r_hv, r_ratio, p("rgb_before.png"))
    if not a.no_png:
        save_rgb(hh_a, hv_a, ok, r_hh, r_hv, r_ratio, p("rgb_after.png"))
    if not a.no_png:
        save_change(d_hh, ok, a.clip, p("change_hh.png"))
    if not a.no_png:
        save_change(d_hv, ok, a.clip, p("change_hv.png"))

    np.savez_compressed(
        p("stack.npz"),
        hh_before=hh_b.astype(np.float32), hv_before=hv_b.astype(np.float32),
        hh_after=hh_a.astype(np.float32), hv_after=hv_a.astype(np.float32),
        change_hh=d_hh.astype(np.float32), change_hv=d_hv.astype(np.float32),
        valid=ok.astype(np.uint8),
    )

    # extent of the OUTPUT image in UTM metres (xCoordinates are treated as pixel centres)
    pix = abs(dx) * b
    x_min = float(xb[c_lo] - abs(dx) / 2)
    y_max = float(yb[r_lo] + abs(dy) / 2)
    x_max = x_min + nc_out * pix
    y_min = y_max - nr_out * pix
    corners = {}
    for name, (x, y) in {"nw": (x_min, y_max), "ne": (x_max, y_max), "sw": (x_min, y_min), "se": (x_max, y_min)}.items():
        lon, lat = to_ll.transform(x, y)
        corners[name] = {"lon": lon, "lat": lat}
    big = (np.abs(d_hv[ok]) > 3).mean() * 100
    info = {
        "before": ib, "after": ia, "epsg": epsg_b, "block": b, "pixel_size_m": pix,
        "utm_extent": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "corners_lonlat": corners, "shape_rows_cols": [nr_out, nc_out],
        "median_shift_db": {"hh": shift_hh, "hv": shift_hv}, "normalised": not a.no_normalise,
        "display_range_db": {"hh": r_hh, "hv": r_hv, "hh_minus_hv": r_ratio},
        "change_clip_db": a.clip, "valid_fraction": float(ok.mean()),
        "percent_abs_change_gt_3db": {"hv": float(big), "hh": float((np.abs(d_hh[ok]) > 3).mean() * 100)},
        "files": {"before": os.path.basename(a.before), "after": os.path.basename(a.after)},
    }
    with open(p("info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"|HV change| > 3 dB on {big:.1f}% of valid pixels")
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
