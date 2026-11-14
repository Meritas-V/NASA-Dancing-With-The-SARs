"""
fetch_weather.py - pull the weather around your two NISAR passes, to help explain the radar change.

Usage (PowerShell, venv active) - reads the two pass times from a process_pair.py run:
  python fetch_weather.py --info out\\overview\\info.json
Optional extras:
  --point "name:lat,lon"   add a place (repeat it). Default places: the detail box centre + the overview centre.
  --days-before 3          how many days before the AFTER pass to include in the daily table (default: whole gap)

Why it matters for radar (the physics, in one paragraph):
  L-band radar is sensitive to water. Wet soil and wet snow change how much energy comes back; frozen ground
  and dry snow behave differently again; rain on the days before a pass raises soil moisture. So when the whole
  scene got ~1 dB darker between Nov 12 and Nov 24, the first questions are: did it freeze? did it snow? did it
  rain a lot just before one of the passes?

Data: Open-Meteo historical archive (reanalysis, ~10-25 km grid cells, NOT a weather station).
      Mountains and valleys inside one cell get the same numbers, so treat it as regional context, not ground truth.
Uses only the Python standard library (no extra install).
"""
import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_CORE = ["temperature_2m", "precipitation", "rain", "snowfall"]
HOURLY_EXTRA = ["snow_depth", "soil_moisture_0_to_7cm"]   # not every model has these -> asked for separately


def fetch(lat, lon, start, end, variables):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(variables), "timezone": "UTC",
    })
    req = urllib.request.Request(URL + "?" + q, headers={"User-Agent": "nisar-spaceapps-project"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get_place(lat, lon, start, end):
    """Fetch core variables, then try the extras; keep whatever the API gives us."""
    data = fetch(lat, lon, start, end, HOURLY_CORE)
    for var in HOURLY_EXTRA:
        try:
            extra = fetch(lat, lon, start, end, [var])
            if extra.get("hourly", {}).get(var) is not None:
                data["hourly"][var] = extra["hourly"][var]
        except Exception as e:  # variable not available for this point/model - fine
            print(f"   (note: {var} not available here: {e})")
    return data


def num(x):
    return x if isinstance(x, (int, float)) else None


def window(vals, i, n):
    """values for the n hours up to and including index i, without None."""
    lo = max(0, i - n + 1)
    return [v for v in (num(x) for x in vals[lo:i + 1]) if v is not None]


def summarise(h, t_pass):
    times = h["time"]
    # index of the hour at/just before the pass
    key = t_pass.strftime("%Y-%m-%dT%H:00")
    i = times.index(key) if key in times else None
    if i is None:
        return None
    out = {}
    temp24, temp72 = window(h["temperature_2m"], i, 24), window(h["temperature_2m"], i, 72)
    out["temp_at_pass_C"] = num(h["temperature_2m"][i])
    out["temp_24h_min/max_C"] = (min(temp24), max(temp24)) if temp24 else None
    out["hours_below_0C_72h"] = sum(1 for v in temp72 if v <= 0)
    out["precip_24h_mm"] = round(sum(window(h["precipitation"], i, 24)), 1)
    out["precip_72h_mm"] = round(sum(window(h["precipitation"], i, 72)), 1)
    out["snowfall_72h_cm"] = round(sum(window(h["snowfall"], i, 72)), 1)
    if "snow_depth" in h:
        out["snow_depth_at_pass_cm"] = None if num(h["snow_depth"][i]) is None else round(h["snow_depth"][i] * 100, 1)
    if "soil_moisture_0_to_7cm" in h:
        out["soil_moisture_0_7cm"] = num(h["soil_moisture_0_to_7cm"][i])
    return out


def daily_table(h, d0, d1):
    rows = []
    day = d0
    while day <= d1:
        prefix = day.strftime("%Y-%m-%d")
        idx = [k for k, t in enumerate(h["time"]) if t.startswith(prefix)]
        temps = [v for v in (num(h["temperature_2m"][k]) for k in idx) if v is not None]
        rows.append({
            "date": prefix,
            "tmin_C": round(min(temps), 1) if temps else None,
            "tmax_C": round(max(temps), 1) if temps else None,
            "precip_mm": round(sum(v for v in (num(h["precipitation"][k]) for k in idx) if v is not None), 1),
            "snowfall_cm": round(sum(v for v in (num(h["snowfall"][k]) for k in idx) if v is not None), 1),
            "snow_depth_cm_end": (None if "snow_depth" not in h or not idx or num(h["snow_depth"][idx[-1]]) is None
                                  else round(h["snow_depth"][idx[-1]] * 100, 1)),
        })
        day += timedelta(days=1)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info", help="info.json written by process_pair.py (gives both pass times and the map corners)")
    ap.add_argument("--before", help="override: before pass time, e.g. 2025-11-12T03:13")
    ap.add_argument("--after", help="override: after pass time, e.g. 2025-11-24T03:13")
    ap.add_argument("--point", action="append", default=[], help='"name:lat,lon" (repeatable)')
    ap.add_argument("--out", default="out")
    ap.add_argument("--from-json", help="debug: read a saved weather.json instead of calling the API")
    a = ap.parse_args()

    tb, ta = a.before, a.after
    default_pts = [("detail-box", 47.895, 25.38)]
    if a.info:
        info = json.load(open(a.info))
        tb = tb or info["before"]["start"]
        ta = ta or info["after"]["start"]
        c = info["corners_lonlat"]
        lat = (c["nw"]["lat"] + c["se"]["lat"]) / 2
        lon = (c["nw"]["lon"] + c["se"]["lon"]) / 2
        default_pts.append(("window-centre", round(lat, 4), round(lon, 4)))
    if not tb or not ta:
        print("Give --info out\\overview\\info.json, or both --before and --after.")
        sys.exit(1)
    t_b = datetime.fromisoformat(tb[:16])
    t_a = datetime.fromisoformat(ta[:16])

    points = list(default_pts)
    for p in a.point:
        name, rest = p.split(":", 1)
        la, lo = rest.split(",")
        points.append((name, float(la), float(lo)))

    start = (t_b - timedelta(days=4)).strftime("%Y-%m-%d")
    end = (t_a + timedelta(days=1)).strftime("%Y-%m-%d")
    result = {"before_pass_utc": t_b.isoformat(), "after_pass_utc": t_a.isoformat(), "places": {}}

    for name, la, lo in points:
        print(f"\n=== {name}  ({la}, {lo}) ===")
        if a.from_json:
            data = json.load(open(a.from_json))["places"][name]["raw"]
        else:
            try:
                data = get_place(la, lo, start, end)
            except urllib.error.URLError as e:
                print("Could not reach the weather API:", e)
                sys.exit(2)
        h = data["hourly"]
        print(f"grid-cell elevation: {data.get('elevation')} m   (the real ground may be much higher/lower)")
        sb, sa = summarise(h, t_b), summarise(h, t_a)
        print("-- 72 h up to the BEFORE pass", t_b.strftime("%Y-%m-%d %H:%MZ"))
        for k, v in (sb or {}).items():
            print(f"   {k}: {v}")
        print("-- 72 h up to the AFTER pass", t_a.strftime("%Y-%m-%d %H:%MZ"))
        for k, v in (sa or {}).items():
            print(f"   {k}: {v}")
        rows = daily_table(h, t_b.replace(hour=0, minute=0), t_a.replace(hour=0, minute=0))
        print("-- daily between the passes")
        print("   date        tmin   tmax   rain/precip mm  snowfall cm  snow depth cm")
        for r in rows:
            print(f"   {r['date']}  {str(r['tmin_C']):>5}  {str(r['tmax_C']):>5}  {r['precip_mm']:>14}  {r['snowfall_cm']:>11}  {str(r['snow_depth_cm_end']):>12}")
        result["places"][name] = {"lat": la, "lon": lo, "elevation_m": data.get("elevation"),
                                  "before": sb, "after": sa, "daily": rows, "raw": data}

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "weather.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    with open(os.path.join(a.out, "weather_daily.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["place", "date", "tmin_C", "tmax_C", "precip_mm", "snowfall_cm", "snow_depth_cm_end"])
        for name, pl in result["places"].items():
            for r in pl["daily"]:
                w.writerow([name, r["date"], r["tmin_C"], r["tmax_C"], r["precip_mm"], r["snowfall_cm"], r["snow_depth_cm_end"]])
    print(f"\nsaved {os.path.join(a.out, 'weather.json')} and weather_daily.csv")


if __name__ == "__main__":
    main()
