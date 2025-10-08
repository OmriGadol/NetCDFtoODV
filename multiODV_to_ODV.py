#!/usr/bin/env python3
"""
Robust ODV merger for mixed CTD text files.

- Reads .txt with sep=None (auto-scan delimiter) and skips only lines starting with //
- Normalizes headers & aliases (TEMP/TEMPS*, PSAL*, PRES/Pressure*, DOXY*/Oxygen*, ATTNDR*/Beam Attenuation, TURB*/Turbidity, chlorophyll *_UGPL)
- Converts BLANK STRINGS to NaN in meta/time columns so forward-fill actually works
- Forward-fills station metadata (Cruise/Station/Type/Lon/Lat/BotDepth + Date/Time)
  -- per station group (Cruise+Station) -- then derives Year/Month/Day/Hour/Minute/Second
- If Depth [m] missing, creates it from Pressure [dbar] (≈1:1)
- Drops obvious junk columns but NEVER drops core variables
- Outputs one ODV Spreadsheet .txt
"""

import os
import io
import glob
import datetime as dt
from typing import List, Optional
import pandas as pd
import numpy as np

# -------------------- USER SETTINGS --------------------
INPUT_DIR   = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/ALLctd/RAW_txt"
OUTPUT_FILE = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/ALLctd/RAW_txt/ctd_merged_odv.txt"

INCLUDE_SOURCEFILE_IN_OUTPUT = False
DEFAULT_TYPE = "CTD"

# If you want to derive attenuation from transmission, set the pathlength (m) of your transmissometer.
# Common values are 0.25 m (Sea-Bird/Chelsea). Set to None to skip conversion.
TRANSM_PATHLENGTH_M: Optional[float] = None
# -------------------------------------------------------

META_ORDER = [
    "Cruise", "Station", "Type",
    "Longitude [degrees_east]", "Latitude [degrees_north]",
    "Year", "Month", "Day", "Hour", "Minute", "Second",
    "Bot. Depth [m]"
]

CANONICAL = {
    "cruise": "Cruise",
    "station": "Station",
    "type": "Type",

    "longitude": "Longitude [degrees_east]",
    "longitude [deg_east]": "Longitude [degrees_east]",
    "lon": "Longitude [degrees_east]",
    "lon (°e)": "Longitude [degrees_east]",
    "lon (?e)": "Longitude [degrees_east]",
    "lon (deg e)": "Longitude [degrees_east]",

    "latitude": "Latitude [degrees_north]",
    "lat": "Latitude [degrees_north]",
    "lat (°n)": "Latitude [degrees_north]",
    "lat (?n)": "Latitude [degrees_north]",
    "lat (deg n)": "Latitude [degrees_north]",

    "bot. depth [m]": "Bot. Depth [m]",
    "bottom depth [m]": "Bot. Depth [m]",
    "depth [m]": "Depth [m]",

    "date/time": "Date/Time",
    "date_time": "Date/Time",
    "date-time": "Date/Time",
    "yyyy-mm-ddthh:mm:ss.sss": "Date/Time",
    "yyyy-mm-dd hh:mm:ss.sss": "Date/Time",

    "pressure [db]": "Pressure [dbar]",
    "temperature [c]": "Temperature [degree_C]",
    "dissolved oxygen [umol/kg]": "Oxygen [umol/kg]",

    # Common mislabel—treat as transmission:
    "beam attenuation (%)": "Beam Transmission [%]",
    "beam transmission (%)": "Beam Transmission [%]",
}

# Exact-name aliases → canonical
MEAS_EXACT = {
    # Core CTD
    "temp": "Temperature [degree_C]",
    "temperature": "Temperature [degree_C]",
    "temperature [deg c]": "Temperature [degree_C]",
    "temperature [°c]": "Temperature [degree_C]",

    "psal": "Salinity [psu]",
    "sal": "Salinity [psu]",
    "salinity": "Salinity [psu]",

    "pres": "Pressure [dbar]",
    "pressure": "Pressure [dbar]",
    "pressure [db]": "Pressure [dbar]",

    "doxy": "Oxygen [umol/kg]",
    "oxygen": "Oxygen [umol/kg]",
    "dissolved oxygen": "Oxygen [umol/kg]",

    # Beam attenuation & transmission
    "attndr01_uprm": "Beam Attenuation [1/m]",
    "beam attenuation [1/m]": "Beam Attenuation [1/m]",
    "beam transmission [%]": "Beam Transmission [%]",

    # Turbidity (NTU)
    "turbpr01_ustu": "Turbidity [NTU]",
    "turbxxxx_ustu": "Turbidity [NTU]",
}

# Prefix aliases → canonical (matches if column name startswith the key)
MEAS_PREFIX = [
    # Core CTD
    ("temps",  "Temperature [degree_C]"),
    ("temp",   "Temperature [degree_C]"),
    ("psal",   "Salinity [psu]"),
    ("salin",  "Salinity [psu]"),
    ("pres",   "Pressure [dbar]"),
    ("press",  "Pressure [dbar]"),
    ("doxy",   "Oxygen [umol/kg]"),
    ("oxygen", "Oxygen [umol/kg]"),
    # Beam attenuation / transmission
    ("attndr", "Beam Attenuation [1/m]"),
    ("beam transmiss", "Beam Transmission [%]"),
    # Turbidity
    ("turb",   "Turbidity [NTU]"),
    # Chlorophyll concentration in µg/L (→ mg/m³)
    ("chlflp", "Fluorescence [mg/m3]"),
    ("cphlpr", "Fluorescence [mg/m3]"),
    ("cphlps", "Fluorescence [mg/m3]"),
    ("cphl",   "Fluorescence [mg/m3]"),
]

# “admin/meta” columns to drop; everything numeric that isn’t obviously useful is auto-filtered later
JUNK_NAMES = {"gobjects", "files", "cruisereports", "patches", "view", "web", "misc", "odv", "mon"}

DATE_COL = "Date/Time"
TIME_PARTS = ["Year", "Month", "Day", "Hour", "Minute", "Second"]

# ----------------- helpers -----------------
def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().replace("\ufeff", "").lower().split())

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = normalize_name(c)
        mapping[c] = CANONICAL.get(key, c)
    return df.rename(columns=mapping)

def blanks_to_nan(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Turn empty strings ('' or whitespace) into NaN so ffill works."""
    if cols is None:
        cols = df.columns.tolist()
    mask = df[cols].select_dtypes(include="object")
    if not mask.empty:
        df[mask.columns] = mask.replace(r"^\s*$", np.nan, regex=True)
    return df

def _coalesce_into(df: pd.DataFrame, target: str, sources: list):
    if not sources:
        return
    if target not in df.columns:
        df[target] = np.nan
    for col in sources:
        if col == target or col not in df.columns:
            continue
        df[target] = df[target].where(
            df[target].notna() & (df[target].astype(str).str.strip() != ""),
            df[col]
        )
    for col in sources:
        if col != target and col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

def unify_measure_columns(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "Temperature [degree_C]": [],
        "Salinity [psu]": [],
        "Pressure [dbar]": [],
        "Oxygen [umol/kg]": [],
        # new groups
        "Beam Attenuation [1/m]": [],
        "Beam Transmission [%]": [],
        "Turbidity [NTU]": [],
        "Fluorescence [mg/m3]": [],  # chlorophyll concentration in mg/m³ (µg/L)
    }
    for c in list(df.columns):
        cn = normalize_name(c)

        # Chlorophyll concentration detection by unit suffix in original name
        # If the raw header ends with _UGPL, it’s µg/L → mg/m³; include in group
        if cn.endswith("_ugpl"):
            groups["Fluorescence [mg/m3]"].append(c)
            continue

        if cn in MEAS_EXACT:
            groups[MEAS_EXACT[cn]].append(c); continue

        for pref, target in MEAS_PREFIX:
            if cn.startswith(pref):
                groups[target].append(c); break

    # Coalesce
    for target, cols in groups.items():
        if cols:
            ordered = [x for x in cols if x == target] + [x for x in cols if x != target]
            _coalesce_into(df, target, ordered)
    return df

def drop_empty_depth_rows(df: pd.DataFrame) -> pd.DataFrame:
    for depth_key in ("Depth [m]", "Depth [meter]", "Depth"):
        if depth_key in df.columns:
            mask = df[depth_key].astype(str).str.strip() != ""
            return df[mask]
    return df

def ensure_depth_from_pressure(df: pd.DataFrame) -> pd.DataFrame:
    if "Depth [m]" not in df.columns and "Pressure [dbar]" in df.columns:
        df["Depth [m]"] = df["Pressure [dbar]"]
    return df

def build_datetime_from_parts(df: pd.DataFrame) -> Optional[pd.Series]:
    md = next((c for c in df.columns if normalize_name(c) == "mon/day/yr"), None)
    hm = next((c for c in df.columns if normalize_name(c) in ("hh:mm", "time")), None)
    if md and hm:
        return (df[md].fillna("").astype(str).str.strip() + " " +
                df[hm].fillna("").astype(str).str.strip())
    date_col = next((c for c in df.columns if normalize_name(c) == "date"), None)
    time_col = next((c for c in df.columns if normalize_name(c) == "time"), None)
    if date_col and time_col:
        return (df[date_col].fillna("").astype(str).str.strip() + " " +
                df[time_col].fillna("").astype(str).str.strip())
    def find(col):
        for c in df.columns:
            if normalize_name(c) == col: return c
        return None
    y, m, d = find("year"), find("month"), find("day")
    if y and m and d:
        h, mi, s = find("hour"), find("minute"), find("second")
        parts = (
            df[y].astype(str).str.zfill(4) + "-" +
            df[m].astype(str).str.zfill(2) + "-" +
            df[d].astype(str).str.zfill(2) + " " +
            (df[h].astype(str).str.zfill(2) if h else "00") + ":" +
            (df[mi].astype(str).str.zfill(2) if mi else "00") + ":" +
            (df[s].astype(str).str.zfill(2) if s else "00")
        )
        return parts
    return None

def parse_datetime_series(series: pd.Series) -> pd.Series:
    def try_parse(t):
        if pd.isna(t): return pd.NaT
        s = str(t).strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%dT%H:%M:%S",     "%Y-%m-%d %H:%M:%S",
                    "%m/%d/%Y %H:%M:%S",     "%m/%d/%Y %H:%M",
                    "%d/%m/%Y %H:%M:%S",     "%d/%m/%Y %H:%M",
                    "%Y-%m-%d",              "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return dt.datetime.strptime(s, fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT
    return series.map(try_parse)

def ensure_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in df.columns:
        combined = build_datetime_from_parts(df)
        if combined is not None:
            df[DATE_COL] = combined
    if DATE_COL in df.columns:
        parsed = parse_datetime_series(df[DATE_COL])
        df["Year"]   = parsed.dt.year
        df["Month"]  = parsed.dt.month.fillna(0).astype(int)
        df["Day"]    = parsed.dt.day.fillna(0).astype(int)
        df["Hour"]   = parsed.dt.hour.fillna(0).astype(int)
        df["Minute"] = parsed.dt.minute.fillna(0).astype(int)
        df["Second"] = parsed.dt.second.fillna(0).astype(int)
    else:
        for col in TIME_PARTS:
            if col not in df.columns:
                df[col] = None
    return df

def ensure_meta_defaults(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "Cruise" not in df.columns:
        df["Cruise"] = os.path.basename(os.path.dirname(source_name)) or os.path.basename(source_name)
    if "Station" not in df.columns:
        df["Station"] = 1
    if "Type" not in df.columns:
        df["Type"] = DEFAULT_TYPE
    return df

def forward_fill_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta_to_ffill = [c for c in META_ORDER if c in df.columns]
    if DATE_COL in df.columns:
        meta_to_ffill.append(DATE_COL)
    # Convert blanks to NaN so ffill fills them
    df = blanks_to_nan(df, cols=[c for c in meta_to_ffill if c in df.columns])
    if meta_to_ffill:
        df[meta_to_ffill] = df[meta_to_ffill].ffill()
    return df

def drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = set(df.columns)
    for c in list(df.columns):
        if normalize_name(c) in JUNK_NAMES:
            keep.discard(c)

    # protect all time-related columns so they don't get dropped
    protected_time_raw = {"mon/day/yr", "hh:mm", "time", "date"}
    protected_time_cols = {c for c in df.columns if normalize_name(c) in protected_time_raw}
    meta_like = set(META_ORDER + [DATE_COL, "Depth [m]"]) | protected_time_cols

    # Always keep our canonical/merged measurement targets
    core_keep = {
        "Pressure [dbar]", "Temperature [degree_C]", "Salinity [psu]", "Oxygen [umol/kg]",
        "Beam Attenuation [1/m]", "Beam Transmission [%]", "Turbidity [NTU]", "Fluorescence [mg/m3]"
    }

    for c in df.columns:
        if c in meta_like or c in core_keep:
            continue
        s = df[c]
        if s.dtype == object:
            vals = s.astype(str).str.strip()
            isnum = vals.str.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
            if isnum.mean() < 0.40:
                keep.discard(c)
    return df.loc[:, list(keep)]

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].str.strip()
    return df

# ----------------- main reader -----------------
def load_one_txt(path: str) -> pd.DataFrame:
    # keep lines; skip only '//' headers (not single '/')
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        kept = [line for line in fh if not line.lstrip().startswith("//")]
    buf = io.StringIO("".join(kept))

    try:
        df = pd.read_csv(buf, sep=None, engine="python", dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"⚠️  Could not read {os.path.basename(path)}: {e}")
        return pd.DataFrame()

    if df.empty or len(df.columns) < 2:
        print(f"⚠️  Skipping (empty): {os.path.basename(path)}")
        return pd.DataFrame()

    df = strip_whitespace(df)
    df = canonicalize_columns(df)
    df = df[[c for c in df.columns if not c.startswith("QV:")]]

    # Unify measurement names first (collects *_UGPL chl, ATTNDR*, TURB*, etc.)
    df = unify_measure_columns(df)

    # SDN short codes or simplified aliases → canonical (safety pass)
    for col in list(df.columns):
        n = normalize_name(col)
        if n == "pres":
            df.rename(columns={col: "Pressure [dbar]"}, inplace=True)
        elif n == "psal":
            df.rename(columns={col: "Salinity [psu]"}, inplace=True)
        elif n == "temp":
            df.rename(columns={col: "Temperature [degree_C]"}, inplace=True)
        elif n.startswith("doxy"):
            df.rename(columns={col: "Oxygen [umol/kg]"}, inplace=True)

    # Coerce canonical numeric variables
    for must in [
        "Pressure [dbar]", "Temperature [degree_C]", "Salinity [psu]", "Oxygen [umol/kg]",
        "Beam Attenuation [1/m]", "Beam Transmission [%]", "Turbidity [NTU]", "Fluorescence [mg/m3]"
    ]:
        if must in df.columns:
            df[must] = pd.to_numeric(df[must], errors="coerce")

    # >>> Build/parse Date/Time BEFORE dropping columns (keeps mon/day/yr + hh:mm)
    df = ensure_time_parts(df)

    # Optional: derive attenuation from transmission if pathlength known
    if TRANSM_PATHLENGTH_M and "Beam Transmission [%]" in df.columns:
        t = pd.to_numeric(df["Beam Transmission [%]"], errors="coerce")
        with np.errstate(invalid="ignore", divide="ignore"):
            c = -(1.0 / TRANSM_PATHLENGTH_M) * np.log(t / 100.0)
        if "Beam Attenuation [1/m]" in df.columns:
            df["Beam Attenuation [1/m]"] = df["Beam Attenuation [1/m]"].fillna(c)
        else:
            df["Beam Attenuation [1/m]"] = c

    # Now it’s safe to remove junk columns
    df = drop_junk_columns(df)

    # defaults
    df = ensure_meta_defaults(df, path)

    # ---- CRITICAL: make blanks NaN in meta/time columns so ffill works ----
    meta_like_cols = [c for c in META_ORDER if c in df.columns]
    if DATE_COL in df.columns:
        meta_like_cols.append(DATE_COL)
    df = blanks_to_nan(df, cols=meta_like_cols)

    # group-wise fill Date/Time and meta
    group_keys = [k for k in ["Cruise", "Station"] if k in df.columns]
    if DATE_COL in df.columns and group_keys:
        df[DATE_COL] = (
            df.groupby(group_keys, dropna=False)[DATE_COL]
              .transform(lambda s: s.ffill().bfill())
        )
    df = forward_fill_meta(df)

    # derive Y/M/D/H/M/S from the filled Date/Time
    df = ensure_time_parts(df)

    # ensure depth
    df = ensure_depth_from_pressure(df)
    df = drop_empty_depth_rows(df)

    if INCLUDE_SOURCEFILE_IN_OUTPUT:
        df["SourceFile"] = os.path.basename(path)

    return df

# ----------------- merge driver -----------------
def main():
    files: List[str] = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.txt"), recursive=True))
    if not files:
        print(f"No .txt files found under: {INPUT_DIR}")
        return

    frames = []
    for p in files:
        df = load_one_txt(p)
        if df.empty:
            continue
        frames.append(df)
        print(f"✓ Loaded {os.path.basename(p)} ({len(df)} rows)")

    if not frames:
        print("No readable files.")
        return

    # union of columns (meta first)
    all_cols, seen = [], set()
    for col in META_ORDER:
        if any(col in f.columns for f in frames):
            all_cols.append(col); seen.add(col)
    if any(DATE_COL in f.columns for f in frames) and DATE_COL not in seen:
        all_cols.append(DATE_COL); seen.add(DATE_COL)
    for f in frames:
        for col in f.columns:
            if col not in seen:
                all_cols.append(col); seen.add(col)

    frames = [f.reindex(columns=all_cols) for f in frames]
    merged = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    merged.to_csv(OUTPUT_FILE, sep="\t", index=False, na_rep="")
    print(f"\n✅ Merged {len(files)} file(s) → {OUTPUT_FILE}\n"
          f"👉 Import into ODV via: Import → ODV Spreadsheet…")

if __name__ == "__main__":
    main()
