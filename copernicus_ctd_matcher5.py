import os
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
"""
===============================================================================
Script: generate_model_only_fullODV.py

Description:
    Reads a CTD ODV text file and Copernicus model NetCDF outputs,
    interpolates temperature (thetao) and salinity (so) at each CTD
    station's location and depths, and writes a single combined ODV-
    compatible output file containing only the model data for each
    station. Supports optional manual stations.

Usage:
    python generate_model_only_fullODV.py

Configuration:
    - Toggle `use_ctd` and `use_manual` in the USER OPTIONS section.
    - Define file paths and optional manual_stations as needed.

===============================================================================
"""
# -------------------------------
# 1. File paths
# -------------------------------
ctd_file    = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/haifa_uni_05.txt'
temp_nc     = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/med-cmcc-tem-rean-m_1744206250724.nc'
sal_nc      = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/med-cmcc-sal-rean-m_1744205630998.nc'
output_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/Coper_model_only.txt'
out_dir = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/'
# -------------------------------
# ====================================================
#                USER OPTIONS
# ====================================================
use_ctd    = True    # If True, pull stations from the CTD file
use_manual = False  # If True, append stations from manual_stations list

# Define manual stations here if use_manual=True
# Each dict must have:
#   'Station', 'Cruise', 'LOCAL_CDI_ID',
#   'datetime' (string, same format as CTD file),
#   'Latitude [degrees_north]', 'Longitude [degrees_east]',
#   'PRESSURES' (list of dbar levels)
manual_stations = [
    # Example:
    # {
    #   'Station': 'MAN1',
    #   'Cruise': 'CustomCast1',
    #   'LOCAL_CDI_ID': 'Manual001',
    #   'datetime': '2013-10-23T07:00:00.000',
    #   'Latitude [degrees_north]': 32.85,
    #   'Longitude [degrees_east]': 35.02,
    #   'PRESSURES': [1, 10, 20, 50, 100]
    # },
     #{
     #  'Station': 'MAN9',
     # 'Cruise': 'CustomCast1',
     #  'LOCAL_CDI_ID': 'Manual001',
      # 'datetime': '2014-10-23T07:00:00.000',
     #  'Latitude [degrees_north]': 32.65,
     #  'Longitude [degrees_east]': 34.80,
     #  'PRESSURES': [1, 10, 20, 50, 100, 120, 150, 200, 220]
     #},
]

# ====================================================
#        READ ORIGINAL ODV HEADER & COLUMNS
# ====================================================
with open(ctd_file, 'r') as f:
    lines = f.readlines()

hdr_idx    = next(i for i, L in enumerate(lines) if L.startswith("Cruise"))
header     = lines[:hdr_idx]
col_header = lines[hdr_idx].rstrip("\n")
cols       = col_header.split("\t")

# column indices to overwrite
i_cruise    = cols.index("Cruise")
i_local_cdi = cols.index("LOCAL_CDI_ID")
i_psal      = cols.index("PSAL")
i_temp      = cols.index("TEMP")

# ====================================================
#            LOAD & CLEAN CTD DATA
# ====================================================
raw = pd.read_csv(ctd_file, sep="\t", skiprows=hdr_idx)
ctd = raw.ffill()
ctd["datetime_parsed"] = pd.to_datetime(ctd["yyyy-mm-ddThh:mm:ss.sss"])

# ====================================================
#        IDENTIFY EACH CTD CAST START
# ====================================================
starts = raw.index[raw["Type"] == "C"].tolist()
starts.append(len(ctd))  # sentinel

# ====================================================
#        OPEN MODEL DATASETS & RANGES
# ====================================================
ds_t = xr.open_dataset(temp_nc)
ds_s = xr.open_dataset(sal_nc)

t_min, t_max       = ds_t.time.min().values, ds_t.time.max().values
lat_min, lat_max   = ds_t.latitude.min().values, ds_t.latitude.max().values
lon_min, lon_max   = ds_t.longitude.min().values, ds_t.longitude.max().values

lats = ds_t["latitude"].values
lons = ds_t["longitude"].values

# ====================================================
#               INTERPOLATION HELPERS
# ====================================================
def bilinear_w(x, y, x0, x1, y0, y1):
    dx, dy = x1 - x0, y1 - y0
    return {
        "ll": (x1 - x) * (y1 - y) / (dx * dy),
        "lr": (x - x0) * (y1 - y) / (dx * dy),
        "ul": (x1 - x) * (y - y0) / (dx * dy),
        "ur": (x - x0) * (y - y0) / (dx * dy),
    }

def sample(ds, var, t, lat, lon, pres):
    return float(
        ds[var]
        .sel(time=t, latitude=lat, longitude=lon, depth=pres, method="nearest")
        .values
    )

def nan_interp(vals, wts):
    valid = {k: wts[k] for k, v in vals.items() if not np.isnan(v)}
    if not valid:
        return np.nan
    W = sum(valid.values())
    return sum(vals[k] * valid[k] for k in valid) / W

# ====================================================
#           PROCESS & WRITE OUTPUT FILE
# ====================================================
with open(output_file, "w") as outf:
    # 1) write header and column line
    outf.writelines(header)
    outf.write(col_header + "\n")

    # helper to process a single cast (CTD or manual)
    def process_cast_block(df_block=None, manual_meta=None):
        # If manual_meta is provided, build a temp DataFrame
        if manual_meta:
            rows = []
            for pres in manual_meta["PRESSURES"]:
                row = {
                    "Cruise": f"Model_{manual_meta['Cruise']}",
                    "Station": manual_meta["Station"],
                    "Type": "C",
                    "yyyy-mm-ddThh:mm:ss.sss": manual_meta["datetime"],
                    "Longitude [degrees_east]": manual_meta["Longitude [degrees_east]"],
                    "Latitude [degrees_north]": manual_meta["Latitude [degrees_north]"],
                    "LOCAL_CDI_ID": f"Model_{manual_meta['LOCAL_CDI_ID']}",
                    "EDMO_code": "",
                    "Bot. Depth [m]": "",
                    "PRES": pres
                }
                # zero out any QV columns
                for c in cols:
                    if c.startswith("QV:"):
                        row[c] = 0
                rows.append(row)
            df = pd.DataFrame(rows)
            df["datetime_parsed"] = pd.to_datetime(df["yyyy-mm-ddThh:mm:ss.sss"])
        else:
            df = df_block.copy()
            # prefix Cruise & LOCAL_CDI_ID for every CTD‑derived row
            df["Cruise"] = df["Cruise"].apply(lambda x: f"Model_{x}")
            df["LOCAL_CDI_ID"] = df["LOCAL_CDI_ID"].apply(lambda x: f"Model_{x}")
        # validate time & coords
        t0 = np.datetime64(df.loc[0, "datetime_parsed"])
        if not (t_min <= t0 <= t_max):
            raise ValueError(f"Date {t0} out of model time range")
        lat0 = df.loc[0, "Latitude [degrees_north]"]
        lon0 = df.loc[0, "Longitude [degrees_east]"]
        if not (lat_min <= lat0 <= lat_max and lon_min <= lon0 <= lon_max):
            raise ValueError(f"Coords ({lat0},{lon0}) out of model domain")

        # nearest model time & grid cell
        t_mod = ds_t["time"].sel(time=t0, method="nearest").values
        ilat  = np.searchsorted(lats, lat0)
        ilon  = np.searchsorted(lons, lon0)
        y0, y1 = lats[ilat - 1], lats[ilat]
        x0, x1 = lons[ilon - 1], lons[ilon]

        # interpolate each depth
        for _, row in df.iterrows():
            pres = row["PRES"]
            wts  = bilinear_w(lon0, lat0, x0, x1, y0, y1)

            Tvals = {
                "ll": sample(ds_t, "thetao", t_mod, y0, x0, pres),
                "lr": sample(ds_t, "thetao", t_mod, y0, x1, pres),
                "ul": sample(ds_t, "thetao", t_mod, y1, x0, pres),
                "ur": sample(ds_t, "thetao", t_mod, y1, x1, pres),
            }
            Svals = {
                "ll": sample(ds_s, "so", t_mod, y0, x0, pres),
                "lr": sample(ds_s, "so", t_mod, y0, x1, pres),
                "ul": sample(ds_s, "so", t_mod, y1, x0, pres),
                "ur": sample(ds_s, "so", t_mod, y1, x1, pres),
            }

            Tmod = nan_interp(Tvals, wts)
            Smod = nan_interp(Svals, wts)

            out = []
            for k, col in enumerate(cols):
                if k == i_cruise:
                    out.append(row["Cruise"])
                elif k == i_local_cdi:
                    out.append(row["LOCAL_CDI_ID"])
                elif k == i_psal:
                    out.append(f"{Smod:.4f}")
                elif k == i_temp:
                    out.append(f"{Tmod:.4f}")
                else:
                    val = row.get(col, "")
                    out.append("" if pd.isna(val) else str(val))
            outf.write("\t".join(out) + "\n")

    # 2) process CTD file casts
    if use_ctd:
        for idx in range(len(starts) - 1):
            block = ctd.iloc[starts[idx] : starts[idx + 1]].reset_index(drop=True)
            process_cast_block(df_block=block)

    # 3) process manual stations
    if use_manual:
        for meta in manual_stations:
            process_cast_block(manual_meta=meta)

print("Wrote combined ODV file to:", output_file)