"""
inspect_gcov.py - open ONE NISAR GCOV .h5 file and print what's inside.

Usage:   python inspect_gcov.py "M:\\path\\to\\file.h5"

What it does (in order):
  1. Prints the layout of the file (groups/datasets, shapes, dtypes) - the "folder tree".
  2. Prints identification info: track, frame, pass direction, start time.
  3. Prints the map grid: projection (EPSG), pixel spacing, bounds, size in km.
  4. Reads a coarse sample of HH and HV (every 20th pixel, so it's fast) and
     prints basic stats in dB, so we know the data is sane before building on it.
Nothing is modified; the file is opened read-only.
"""
import sys
import numpy as np
import h5py

GRID = "/science/LSAR/GCOV/grids/frequencyA"
IDENT = "/science/LSAR/identification"


def show_tree(f, max_items=400):
    print("=" * 70)
    print("FILE LAYOUT (name  shape  dtype)")
    print("=" * 70)
    count = 0

    def visit(name, obj):
        nonlocal count
        count += 1
        if count > max_items:
            return
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}   {obj.shape}   {obj.dtype}")
        else:
            print(f"[{name}]")

    f.visititems(visit)
    if count > max_items:
        print(f"... ({count - max_items} more items not shown)")


def decode(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray) and v.shape == ():
        return decode(v.item())
    return v


def show_identification(f):
    print("\n" + "=" * 70)
    print("IDENTIFICATION")
    print("=" * 70)
    if IDENT not in f:
        print("  (no identification group found at", IDENT, ")")
        return
    grp = f[IDENT]
    wanted = ["trackNumber", "frameNumber", "orbitPassDirection",
              "absoluteOrbitNumber", "zeroDopplerStartTime", "zeroDopplerEndTime",
              "productType", "boundingPolygon"]
    for key in wanted:
        if key in grp:
            try:
                print(f"  {key}: {decode(grp[key][()])}")
            except Exception as e:
                print(f"  {key}: <could not read: {e}>")
        else:
            print(f"  {key}: (not present)")


def show_grid(f):
    print("\n" + "=" * 70)
    print("MAP GRID (frequency A)")
    print("=" * 70)
    if GRID not in f:
        print("  (no grid group at", GRID, ")")
        return None
    g = f[GRID]
    x = g["xCoordinates"][()]
    y = g["yCoordinates"][()]
    print(f"  pixels: {len(x)} columns x {len(y)} rows")
    print(f"  x range: {x.min():.1f} .. {x.max():.1f}")
    print(f"  y range: {y.min():.1f} .. {y.max():.1f}")
    for key in ("xCoordinateSpacing", "yCoordinateSpacing"):
        if key in g:
            print(f"  {key}: {decode(g[key][()])}")
    epsg = None
    if "projection" in g:
        p = g["projection"]
        try:
            epsg = decode(p[()])
            print(f"  projection value: {epsg}")
        except Exception:
            pass
        for k, v in p.attrs.items():
            sv = str(decode(v))
            print(f"  projection attr {k}: {sv[:120]}")
    dx = abs(float(x[1] - x[0])) if len(x) > 1 else float("nan")
    dy = abs(float(y[1] - y[0])) if len(y) > 1 else float("nan")
    print(f"  measured pixel size: {dx:.1f} x {dy:.1f} (map units, normally metres)")
    print(f"  approx. size: {(x.max()-x.min())/1000:.0f} km x {(y.max()-y.min())/1000:.0f} km")
    return g


def sample_stats(g, step=20):
    print("\n" + "=" * 70)
    print(f"QUICK LOOK (every {step}th pixel)")
    print("=" * 70)
    for pol in ("HHHH", "HVHV"):
        if pol not in g:
            print(f"  {pol}: (not present)")
            continue
        ds = g[pol]
        sample = ds[::step, ::step].astype(np.float64)
        valid = np.isfinite(sample) & (sample > 0)
        frac = valid.mean() * 100
        if valid.any():
            db = 10.0 * np.log10(sample[valid])
            p2, p50, p98 = np.percentile(db, [2, 50, 98])
            print(f"  {pol}: full shape {ds.shape}, valid pixels {frac:.1f}%")
            print(f"     dB  2%={p2:.1f}  median={p50:.1f}  98%={p98:.1f}")
        else:
            print(f"  {pol}: no valid pixels in sample")
    if "mask" in g:
        m = g["mask"][::step, ::step]
        vals, counts = np.unique(m, return_counts=True)
        print("  mask values (sample):", dict(zip(vals.tolist(), counts.tolist())))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        show_tree(f)
        show_identification(f)
        g = show_grid(f)
        if g is not None:
            sample_stats(g)
    print("\nDone.")


if __name__ == "__main__":
    main()
