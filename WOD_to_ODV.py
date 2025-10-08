


#!/usr/bin/env python3
"""
wod_to_odv_spreadsheet_union.py  —  PyCharm-ready (two-pass)

Build ONE ODV Spreadsheet (.txt) from a folder of WOD NetCDF files.
- Pass 1: scan ALL files to discover the union of profile variables on 'z'
  (e.g., Temperature, Salinity, Oxygen, Pressure, ...), excluding flags/sigfigs.
- Always include Depth from 'z' (even if it's a coordinate).
- Pass 2: write rows for every file, filling blanks for missing vars.
- Robust WOD time decoding (e.g., 'days since 1770-01-01 00:00:00').

Import result into ODV: Import → ODV Spreadsheet…
"""

import os
import glob
import math
import datetime as dt
from typing import Optional, List, Tuple, Dict

import numpy as np
import xarray as xr

## -------------------------------------------------------------------------
# 🔧 USER SETTINGS (edit these before running)
INPUT_DIR = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Cadiz/Oceanographic/DataSelection_20251001_123413_15470572"   # full path to folder with .nc files
OUTPUT_FILE = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Cadiz/Oceanographic/odv_wod_DataSelection_20251001_123413_15470572.txt"  # output ODV spreadsheet
# -------------------------------------------------------------------------

CANDIDATE = {
    "lon": ["lon", "longitude", "longitude [degrees_east]", "xlon", "lon_d", "lon_east"],
    "lat": ["lat", "latitude", "latitude [degrees_north]", "ylat", "lat_d", "lat_north"],
    "time": ["time", "date_time", "datetime", "date"],
    "depth": ["z", "depth", "depth [m]", "depthm", "depth_m", "depth (m)"],
    "bottom_depth": ["bottom_depth", "bottom_depth [meters]", "bot. depth [m]", "bot_depth"],
    "type": ["dataset", "instrument_type", "platform", "profile_type", "inst_type"],
}

def normalize_name(name: str) -> str:
    n = name.lower()
    for ch in "[]()":
        n = n.replace(ch, " ")
    return " ".join(n.split())

def find_var(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    ds_keys = {normalize_name(k): k for k in list(ds.data_vars) + list(ds.coords)}
    for cand in candidates:
        key = normalize_name(cand)
        if key in ds_keys:
            return ds_keys[key]
        for k in ds_keys.values():
            if k.lower() == cand.lower():
                return k
    return None

def _safe_text_from_da(da: xr.DataArray) -> str:
    try:
        arr = da.data
        if hasattr(arr, "compute"):
            arr = arr.compute()
        arr = np.asarray(arr)
    except Exception:
        return str(da)
    flat = arr.ravel()
    if flat.dtype.kind == "S":
        try:
            return b"".join(list(flat)).decode("utf-8", errors="ignore")
        except Exception:
            return str(flat)
    if flat.dtype.kind == "U":
        try:
            return "".join(list(flat))
        except Exception:
            return str(flat)
    try:
        return "".join(
            (x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x))
            for x in flat
        )
    except Exception:
        return str(flat)

def infer_type(ds: xr.Dataset, path_hint: Optional[str] = None) -> str:
    for cand in CANDIDATE["type"]:
        for v in ds.data_vars:
            if cand.lower() == v.lower():
                txt = _safe_text_from_da(ds[v]).upper()
                if "XBT" in txt:
                    return "XBT"
                if "CTD" in txt:
                    return "CTD"
    # salinity present → CTD, otherwise check filename, else XBT
    if any("salinity" == normalize_name(v) for v in ds.variables):
        return "CTD"
    if path_hint:
        u = path_hint.upper()
        if "XBT" in u: return "XBT"
        if "CTD" in u: return "CTD"
    return "XBT"

def manual_decode_time(time_val, units: str) -> Optional[dt.datetime]:
    """Parse 'days since YYYY-MM-DD[ HH:MM:SS]'."""
    try:
        if "since" not in units:
            return None
        origin = units.split("since", 1)[1].strip()
        fmt = "%Y-%m-%d %H:%M:%S" if " " in origin else "%Y-%m-%d"
        origin_dt = dt.datetime.strptime(origin, fmt)
        return origin_dt + dt.timedelta(days=float(time_val))
    except Exception:
        return None

def as_odv_datetime(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S")

def discover_profile_vars_union(files: List[str]) -> Tuple[List[Tuple[str, str]], Tuple[str, str]]:
    """
    Scan all files; return:
      - list of (var_name, units) for union of 1D numeric variables along 'z' (no flags/sigfigs)
      - ('z', units) depth info (from coord or data var). Units may be empty if not provided.
    """
    union: Dict[str, str] = {}
    z_units = ""
    saw_any = False

    for path in files:
        try:
            ds = xr.open_dataset(path, decode_times=False)
        except Exception:
            continue

        # Depth (z) units if available
        if "z" in ds.coords:
            z_units = z_units or ds["z"].attrs.get("units", "")
        elif "z" in ds.variables:
            z_units = z_units or ds["z"].attrs.get("units", "")

        # profile variables
        zlen = ds.sizes.get("z", ds.dims.get("z", None))
        for v in ds.variables:
            if v in ds.coords:
                continue
            da = ds[v]
            if "z" in da.dims and da.ndim == 1 and (zlen is None or len(da) == zlen):
                if str(da.dtype).startswith(("float", "int")):
                    lower = v.lower()
                    if any(k in lower for k in ["sigfig", "flag", "wodf", "wodfp", "wodfd"]):
                        continue
                    saw_any = True
                    # keep first non-empty units encountered
                    units = da.attrs.get("units", "")
                    if v not in union or (not union[v] and units):
                        union[v] = units
        ds.close()

    # Sort with a nice order: Temperature, Salinity, Oxygen, Pressure, then others
    preferred = ["temperature", "salinity", "oxygen", "pressure"]
    def sort_key(item):
        name = item[0].lower()
        pref_idx = preferred.index(name) if name in preferred else 999
        return (pref_idx, name)

    prof_vars = sorted(union.items(), key=sort_key)
    return prof_vars, ("z", z_units if z_units else "m")

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.nc"), recursive=True))
    if not files:
        print(f"No .nc files found under: {INPUT_DIR}")
        return

    # PASS 1: discover union of variables + depth units
    prof_vars, z_info = discover_profile_vars_union(files)
    if not prof_vars:
        print("⚠️ No profile variables discovered on 'z' across your files. "
              "Check that these are WOD profile NetCDFs.")
        return

    # Prepare headers
    def odv_col(name: str, units: str) -> str:
        return f"{name} [{units}]" if units else name

    meta_cols = [
        "Cruise", "Station", "Type",
        "Longitude [degrees_east]", "Latitude [degrees_north]",
        "Year", "Month", "Day", "Hour", "Minute", "Second",
        "Bot. Depth [m]"
    ]
    depth_col = odv_col("Depth", z_info[1])  # e.g., Depth [m]
    data_cols = [odv_col(n, u) for (n, u) in prof_vars]
    header = "\t".join(meta_cols + [depth_col] + data_cols)

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n")

        station_counter = 0
        total_rows = 0

        # PASS 2: write rows
        for path in files:
            try:
                ds = xr.open_dataset(path, decode_times=False)
            except Exception as e:
                print(f"SKIP (cannot open): {os.path.basename(path)} -> {e}")
                continue

            lon_name = find_var(ds, CANDIDATE["lon"])
            lat_name = find_var(ds, CANDIDATE["lat"])
            botd_name = find_var(ds, CANDIDATE["bottom_depth"])
            time_name = find_var(ds, CANDIDATE["time"])
            if lon_name is None or lat_name is None:
                print(f"SKIP (missing lat/lon) in {os.path.basename(path)}")
                ds.close()
                continue

            try:
                lon = float(np.atleast_1d(ds[lon_name].to_numpy())[0])
                lat = float(np.atleast_1d(ds[lat_name].to_numpy())[0])
            except Exception:
                print(f"SKIP (bad lat/lon) in {os.path.basename(path)}")
                ds.close()
                continue

            # Depth array (coord or var)
            if "z" in ds.coords:
                z = np.asarray(ds["z"].to_numpy(), dtype=float).ravel()
            elif "z" in ds.variables:
                z = np.asarray(ds["z"].to_numpy(), dtype=float).ravel()
            else:
                print(f"SKIP (no 'z' depth) in {os.path.basename(path)}")
                ds.close()
                continue

            # Build arrays for each discovered variable (pad with NaN if missing)
            arrays = []
            for name, _u in prof_vars:
                if name in ds:
                    arr = np.asarray(ds[name].to_numpy(), dtype=float).ravel()
                else:
                    arr = np.full_like(z, np.nan, dtype=float)
                arrays.append(arr)

            # Align lengths to min length across all arrays + z
            lengths = [len(z)] + [len(a) for a in arrays]
            n = min(lengths)
            if n == 0:
                print(f"SKIP (empty profiles) in {os.path.basename(path)}")
                ds.close()
                continue
            z = z[:n]
            arrays = [a[:n] for a in arrays]

            # Bottom depth
            botd = ""
            if botd_name is not None and botd_name in ds and ds[botd_name].size > 0:
                try:
                    bd = float(np.atleast_1d(ds[botd_name].to_numpy())[0])
                    if math.isfinite(bd):
                        botd = f"{bd:.3f}"
                except Exception:
                    botd = ""

            # Time
            t = None
            if time_name is not None and time_name in ds and ds[time_name].size > 0:
                units = ds[time_name].attrs.get("units", "")
                try:
                    t0 = float(np.atleast_1d(ds[time_name].to_numpy())[0])
                    t = manual_decode_time(t0, units)
                except Exception:
                    pass
            if t is None and "date" in ds.variables:

                try:
                    ival = int(np.atleast_1d(ds["date"].to_numpy())[0])
                    year, month, day = ival // 10000, (ival % 10000) // 100, ival % 100
                    t = dt.datetime(int(year), int(month), int(day))
                except Exception:
                    t = None

            if t is None:
                t = dt.datetime(1970, 1, 1)

            Y, M, D = t.year, t.month, t.day
            h, m, s = t.hour, t.minute, t.second

            # Meta
            typ = infer_type(ds, path_hint=path)
            cruise = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
            station_counter += 1
            station_id = station_counter

            # Write rows
            for i in range(n):
                row_meta = [
                    cruise, str(station_id), typ,
                    f"{lon:.6f}", f"{lat:.6f}",
                    str(Y), str(M), str(D), str(h), str(m), str(s),
                    botd
                ]
                row_depth = [f"{float(z[i]):.6g}" if np.isfinite(z[i]) else ""]
                row_data = []
                for a in arrays:
                    v = a[i]
                    row_data.append("" if not np.isfinite(v) else f"{float(v):.6g}")
                f.write("\t".join(row_meta + row_depth + row_data) + "\n")
                total_rows += 1

            ds.close()

    print(f"✅ Done. Wrote {total_rows} profile rows from {station_counter} file(s) to:\n  {OUTPUT_FILE}")
    print("👉 Import into ODV via: Import → ODV Spreadsheet…")

if __name__ == "__main__":
    main()
