#!/usr/bin/env python3
import os
import glob
import pandas as pd

# -----------------------------------------------------------------------------
# USER CONFIG
# -----------------------------------------------------------------------------
# Folder containing your ODV‐style CTD .txt files
ctd_folder = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/CTDsampels"

# Pattern to match—change if your files are .tab or use different extension
pattern = os.path.join(ctd_folder, "*.txt")

# -----------------------------------------------------------------------------
# PROCESS EACH FILE
# -----------------------------------------------------------------------------
for ctd_file in glob.glob(pattern):
    # --- 1) read header & find where the column line begins
    with open(ctd_file, "r") as f:
        lines = f.readlines()
    hdr_idx    = next(i for i, L in enumerate(lines) if L.startswith("Cruise"))
    col_header = lines[hdr_idx].rstrip("\n")
    cols       = col_header.split("\t")

    # --- 2) load data, forward‐fill missing metadata
    raw = pd.read_csv(ctd_file, sep="\t", skiprows=hdr_idx)
    ctd = raw.ffill()

    # --- 3) auto-detect & rename the columns we need
    lon_col = next(c for c in cols if "Longitude" in c)
    lat_col = next(c for c in cols if "Latitude" in c)
    pres_col = next(c for c in cols if c == "PRES" or "PRESS" in c)
    temp_col = next(c for c in cols if "TEMP" in c and not c.startswith("QV"))
    psal_col = next(c for c in cols if "PSAL" in c and not c.startswith("QV"))
    date_col = next(c for c in cols if "yyyy-mm-dd" in c)

    out = (
        ctd.loc[:, [lon_col, lat_col, pres_col, temp_col, psal_col, date_col]]
        .rename(columns={
            lon_col: "Xcoordinate",
            lat_col: "Ycoordinate",
            pres_col: "Z",
            temp_col: "Temperature",
            psal_col: "Salinity",
            date_col: "Date"
        })
    )

    # --- 4) write out CSV next to the original .txt
    base     = os.path.splitext(os.path.basename(ctd_file))[0]
    out_path = os.path.join(ctd_folder, f"{base}_XYZTS.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
