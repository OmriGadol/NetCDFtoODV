#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge all PANGAEA .tab CTD files from a folder into a single ODV-ready TSV.

✅ Designed to run directly inside PyCharm — just edit INPUT_FOLDER and OUTPUT_FILE.
"""

import os, re, glob
import pandas as pd
from dateutil import parser as dateparser

# ---------- 🔧 USER SETTINGS ----------
INPUT_FOLDER = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Cadiz/Oceanographic/PANGEA"  # 👈 change to your folder
OUTPUT_FILE  = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Cadiz/Oceanographic/PANGEA/odv_merged2.txt"   # 👈 output file location
RECURSE      = False                                   # True if you want to search subfolders
# -------------------------------------

def find_first_table_line(lines):
    for i in range(len(lines)-1):
        header = lines[i].rstrip("\n")
        nxt = lines[i+1].rstrip("\n")
        if len(header.split("\t")) >= 3 and any(re.search(r"[A-Za-z]", c) for c in header):
            nums = sum(bool(re.search(r"[-+]?\d", c)) for c in nxt.split("\t"))
            if nums >= 2:
                return i
    return None

def parse_header_metadata(lines):
    meta = {}
    header_text = "\n".join(lines[:300])
    m = re.search(r"LATITUDE:\s*([-+]?\d+(?:\.\d+)?)", header_text, re.IGNORECASE)
    if m: meta["latitude"] = float(m.group(1))
    m = re.search(r"LONGITUDE:\s*([-+]?\d+(?:\.\d+)?)", header_text, re.IGNORECASE)
    if m: meta["longitude"] = float(m.group(1))
    m = re.search(r"DATE/TIME START:\s*([^\n*]+)", header_text, re.IGNORECASE)
    if m:
        try: meta["datetime"] = dateparser.parse(m.group(1).strip())
        except: pass
    return meta

def sanitize_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    ren = {}
    for c in df.columns:
        low = c.lower()
        if "depth" in low and "[m]" in c:
            ren[c] = "Depth [m]"
        elif "press" in low and "dbar" in low:
            ren[c] = "Press [dbar]"
        elif "temp" in low:
            ren[c] = "Temp [°C]"
        elif "sal" in low:
            ren[c] = "Sal"
        elif "oxygen" in low or "o2" in low:
            ren[c] = "O2 [µmol/kg]"
        elif "sigma" in low:
            ren[c] = "Sigma-theta [kg/m^3]"
        elif "lat" in low:
            ren[c] = "Latitude [degrees_north]"
        elif "lon" in low:
            ren[c] = "Longitude [degrees_east]"
        elif "date" in low or "time" in low:
            ren[c] = "Date/Time"
    return df.rename(columns=ren)

def expand_datetime_column(s):
    def parse_one(x):
        try:
            return dateparser.parse(str(x).strip()) if pd.notna(x) and str(x).strip() else None
        except:
            return None
    dt = s.apply(parse_one)
    return pd.DataFrame({
        "Year": dt.apply(lambda d: d.year if d else pd.NA),
        "Month": dt.apply(lambda d: d.month if d else pd.NA),
        "Day": dt.apply(lambda d: d.day if d else pd.NA),
        "Hour": dt.apply(lambda d: d.hour if d else pd.NA),
        "Minute": dt.apply(lambda d: d.minute if d else pd.NA),
        "Second": dt.apply(lambda d: d.second if d else pd.NA),
    })

def read_pangaea_tab(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    meta = parse_header_metadata(lines)
    hline = find_first_table_line(lines)
    if hline is None:
        raise RuntimeError(f"No table found in {os.path.basename(path)}")
    from io import StringIO
    df = pd.read_csv(StringIO("".join(lines[hline:])), sep="\t", dtype=str).apply(lambda s: s.str.strip())
    df = sanitize_columns(df)

    # depth fallback
    if "Depth [m]" not in df.columns:
        if "Press [dbar]" in df.columns:
            df["Depth [m]"] = pd.to_numeric(df["Press [dbar]"], errors="coerce")
        else:
            df["Depth [m]"] = pd.NA
    else:
        df["Depth [m]"] = pd.to_numeric(df["Depth [m]"], errors="coerce")

    for c in ["Longitude [degrees_east]", "Latitude [degrees_north]", "Temp [°C]", "Sal", "O2 [µmol/kg]"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Latitude [degrees_north]" not in df.columns and "latitude" in meta:
        df["Latitude [degrees_north]"] = meta["latitude"]
    if "Longitude [degrees_east]" not in df.columns and "longitude" in meta:
        df["Longitude [degrees_east]"] = meta["longitude"]

    # date expansion
    if "Date/Time" in df.columns:
        parts = expand_datetime_column(df["Date/Time"])
    else:
        dt = meta.get("datetime", None)
        parts = pd.DataFrame({
            "Year": [dt.year if dt else pd.NA] * len(df),
            "Month": [dt.month if dt else pd.NA] * len(df),
            "Day": [dt.day if dt else pd.NA] * len(df),
            "Hour": [dt.hour if dt else pd.NA] * len(df),
            "Minute": [dt.minute if dt else pd.NA] * len(df),
            "Second": [dt.second if dt else pd.NA] * len(df),
        })
    df = pd.concat([df, parts], axis=1)
    df["Cruise"] = os.path.splitext(os.path.basename(path))[0]
    df["Station"] = df["Cruise"]
    df["Type"] = "C"
    return df

# ---------- MAIN ----------
pattern = "**/*.tab" if RECURSE else "*.tab"
files = glob.glob(os.path.join(INPUT_FOLDER, pattern), recursive=RECURSE)
if not files:
    raise SystemExit("No .tab files found.")

all_df = []
for f in files:
    try:
        df = read_pangaea_tab(f)
        all_df.append(df)
        print(f"✓ {os.path.basename(f)} ({len(df)} rows)")
    except Exception as e:
        print(f"✗ {os.path.basename(f)} - {e}")

merged = pd.concat(all_df, ignore_index=True, sort=False)

# reorder
meta_cols = ["Cruise", "Station", "Type", "Longitude [degrees_east]", "Latitude [degrees_north]",
             "Year", "Month", "Day", "Hour", "Minute", "Second"]
cols = meta_cols + [c for c in merged.columns if c not in meta_cols]
merged = merged[cols]

# save
merged.to_csv(OUTPUT_FILE, sep="\t", index=False)
print(f"\n✅ Merged ODV file written to: {OUTPUT_FILE}\nRows: {len(merged)}")
