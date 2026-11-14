# Dancing with the SARs

A desktop viewer for **surface change between two NISAR radar passes**.
Give it two NISAR L2 GCOV files of the same track and frame, pick an area, and it shows
what got brighter, darker or different between the two dates, with a built-in guide that explains
every colour, unit and caveat, and cites its sources.

Made for the NASA Space Apps Challenge 2026, challenge **"Dancing with the SARs"**.
Open source under the MIT licence. Works for any NISAR GCOV pair, not tied to one location.

<!-- add a screenshot: save it as docs/screenshot.png and uncomment
![Screenshot](docs/screenshot.png)
-->

## What it does

- Reads two NISAR GCOV (L2, gamma-0 power, terrain-corrected) HDF5 files.
- Refuses mismatched pairs (different track, frame, pass direction or grid) with a clear message.
- Uses only pixels the product's mask marks as valid (mask values 1-5, spec sec. 4.3.3).
- Averages in linear power, converts to dB, then maps the change (after minus before).
- Views: change map, before/after swipe, HH/HV composite, statistics, optional satellite basemap, optional weather between the two passes.
- Every colour in the legend is computed from the data actually loaded.
- The Guide (F1) and Terms page explain polarisations, dB, projections, known data issues, with links to the original sources.

## Download (Windows)

Grab the zip from the **Releases** page, unzip, run `DancingWithTheSARs.exe`.
Windows SmartScreen may warn "unknown publisher" because the exe is not code-signed:
click *More info -> Run anyway*.

## Run from source

Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python viewer.py
```

1. **Process pair** (Ctrl+N): choose the BEFORE and AFTER `.h5` files and a lon/lat box.
2. **Open results** (Ctrl+O) opens an earlier result folder.
3. **Weather** downloads reanalysis weather for the two pass times (needs internet).

Command-line equivalent:

```powershell
python process_pair.py BEFORE.h5 AFTER.h5 --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX --block 4 --name myrun
python viewer.py out\myrun
```

## Getting NISAR data

NISAR GCOV files are free from the ASF DAAC (NASA Earthdata login required):
https://search.asf.alaska.edu/ (dataset: NISAR, product: GCOV). Files are large (about 1 GB each);
the app reads only the window you select.

## Build the exe yourself

```powershell
.\build_exe.ps1            # folder + zip in dist\
.\build_exe.ps1 -OneFile   # single exe (slower start)
```

## Data and sources

- **NISAR data**: NASA/JPL and ISRO, distributed by the ASF DAAC. Product spec: JPL D-102274 Rev E.
- **Basemap (optional, downloaded at runtime, not bundled)**: Sentinel-2 cloudless by EOX IT Services GmbH (contains modified Copernicus Sentinel data), https://s2maps.eu, CC BY-NC-SA 4.0 (non-commercial).
- **Weather (optional, downloaded at runtime)**: Open-Meteo (https://open-meteo.com, CC BY 4.0) serving ECMWF ERA5 reanalysis (Hersbach et al. 2020, https://doi.org/10.24381/cds.adbb2d47).
- **Method references**: NASA SAR Handbook ch. 3 (RGB composites); ASF "Introduction to SAR"; Park et al. 2015; Hardaway et al. 1982; Thirion-Lefevre et al. 2020; Santoro et al. 2009. Full list with links in the app's Guide -> References.
- **Libraries**: PySide6 (LGPL), pyqtgraph (MIT), h5py (BSD), pyproj (MIT), NumPy (BSD), Pillow (HPND).

## Limitations

- NISAR GCOV is currently BETA/PROVISIONAL depending on the file: known issues include mask misalignment, banding and geolocation error. The Guide lists them.
- Radar change is not a diagnosis. Moisture, freeze/thaw, vegetation, snow and real surface change can look alike; the app shows where backscatter changed, not why.
- Small pixel-level differences between passes can come from processing, not the ground.

## AI disclosure

The code was written with the help of an AI assistant (Anthropic Claude) under the author's direction;
the author tested the results and chose the design. See the Space Apps submission for details.

## Licence

MIT, see `LICENSE`.
