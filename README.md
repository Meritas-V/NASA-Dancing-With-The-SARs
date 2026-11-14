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

## How to use it, step by step

The app has no data of its own. It needs **two NISAR GCOV files** that you download once, then it processes
them into a small result folder that it can open instantly afterwards.

### 1. Get the data (free)

1. Create a free **Earthdata Login** account: https://urs.earthdata.nasa.gov/users/new
2. Open **Vertex**, the Alaska Satellite Facility (ASF) search site: https://search.asf.alaska.edu/ and sign in (top right).
3. Set *Search Type* to **Geographic Search**, *Dataset* to **NISAR**, and *Science Product* to **GCOV (Geocoded Polarimetric Covariance)**.
4. Draw or zoom to your area and set a date range. Press **Search**.
5. Pick **two acquisitions of the same area** from the same **track**, **frame** and **pass direction** (ascending or descending). The earlier is "before", the later is "after".
   NISAR's track-frame map is on the NISAR mission website. Files from different tracks or frames are refused by the app.
6. For each one, open its file list and download the **HDF5 (.h5)** science file (the download icon under *Science Data*). Each file is about 1 GB. Keep both in one folder.

Other ways to get the same files (Earthdata Search, the `asf_search` or `earthaccess` Python packages) are described in the
ASF NISAR data guide: https://nisar-docs.asf.alaska.edu/access-overview

The product is released as BETA or PROVISIONAL (data maturity), which affects how much to trust fine detail. The in-app Guide explains the known issues.

### 2. Start the app

- **Windows exe:** unzip the release and run `DancingWithTheSARs.exe`. Windows may show an "unknown publisher" warning; choose *More info -> Run anyway*.
- **From source:** see "Run from source" below.

The window opens empty on first start. That is normal.

### 3. Process the pair

Click **Process pair** (Ctrl+N) in the toolbar.

| Field | What to do |
|---|---|
| BEFORE .h5 / AFTER .h5 | Pick the two files (or drag them in). The earlier date becomes BEFORE. |
| Area (degrees) | The window to analyse: lon min, lat min, lon max, lat max (WGS 84). It must lie inside the files' footprint. Only this window is read from disk, so a smaller area is much faster. The fields start with a sample box; **replace it with your own area**. |
| Block size (px) | Averages N x N pixels. One pixel is 20 m for the standard product, so 4 gives 80 m. Higher is smoother and faster, lower is sharper but noisier. |
| Result name | Name of the result folder. |
| Output folder | Where results are written (default: `out` next to the app). |

Press **Process** and watch the log. The app first checks that both files are valid GCOV products and comparable (same track, frame, pass direction, grid).
If not, a message says what is wrong and nothing is processed.

### 4. Explore the result

The result opens automatically. Open an older one any time with **Open results** (Ctrl+O).

- **Views:** change map, before/after swipe, HH/HV composite, each with a legend computed from the loaded data. Zoom with the wheel, pan by dragging.
- **Frame:** toggle the picture frame to see exactly what a screenshot will contain.
- **Satellite photo:** tick the satellite checkbox for a Sentinel-2 basemap (internet needed); an opacity slider appears.
- **Statistics tab:** numbers for the area.
- **Weather:** press the Weather button to download hourly reanalysis weather for the two pass times (internet needed). It helps judge whether a change is real ground change or just wet or frozen ground.
- **Save picture** (Ctrl+S) exports the current view. **Guide** (F1) explains every term, colour and caveat, with sources.

### 5. What happens to your data

1. Only the selected window is read from the two HDF5 files.
2. Pixels the file marks invalid are dropped (only mask values 1 to 5 are used).
3. Power is averaged over each block in **linear** units, then converted to decibels (10 x log10).
4. Change = AFTER dB minus BEFORE dB, with the regional median shift removed so that a uniform offset does not tint the whole map.
5. Results (arrays, preview images and a small `info.json`) are saved in the output folder. Your original files are never modified.

Everything runs on your computer. The only internet use is the optional basemap tiles and weather.

## Run from source

Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python viewer.py
```

Then follow the steps above. Command-line equivalent of "Process pair":

```powershell
python process_pair.py BEFORE.h5 AFTER.h5 --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX --block 4 --name myrun
python viewer.py out\myrun
```

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
