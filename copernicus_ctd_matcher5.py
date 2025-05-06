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
#ctd_file    = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/haifa_uni_05.txt'
ctd_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/2012-2020_UNRESTRICTED/haisec29_bsgas01.txt'
temp_nc     = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/med-cmcc-tem-rean-m_1744206250724.nc'
sal_nc      = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/med-cmcc-sal-rean-m_1744205630998.nc'
#output_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/Coper_model_only.txt'
output_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/haisec29_bsgas01_model_only.txt'
out_dir = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/'
# -------------------------------
# ====================================================
#                USER OPTIONS
# ====================================================
use_ctd    = False    # If True, pull stations from the CTD file
use_manual = False  # If True, append hard-codded stations from manual_stations list
use_manual_csv = True  # If True, read extra stations from CSV file
apply_smoothing        = False  # If True, apply 1D running mean after (or without) vertical interp
apply_vertical_interp  = True   # If True, upsample depths & linearly interpolate vertically
vertical_levels        = 50     # Number of fine depth levels when vertical_interp is True
smoothing_window  = 5      # window size (number of depth points) for running mean




# Define manual stations here if use_manual=True
# Each dict must have:
#   'Station', 'Cruise', 'LOCAL_CDI_ID',
#   'datetime' (string, same format as CTD file),
#   'Latitude [degrees_north]', 'Longitude [degrees_east]',
#   'PRESSURES' (list of dbar levels)
#manual_stations = [
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
#]
 #Path to your CSV of custom points
# CSV format described in README
manual_csv_path = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Batymetry/00048GEOM.csv'
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
#i_psal      = cols.index("PSAL")
#i_temp      = cols.index("TEMP")
i_psal      = cols.index('PSALST01_UPPT')
i_temp      = cols.index("TEMPS901_UPAA")

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

# smoothing helper
def running_mean(x, N):
    """Return N-point running mean of 1D array x (same length)."""
    w = np.ones(N)/N
    return np.convolve(x, w, mode='same')


# ====================================================
#           PROCESS & WRITE OUTPUT FILE
# ====================================================
skipped = []
with open(output_file, "w") as outf:
    # 1) write header and column line
    outf.writelines(header)
    outf.write(col_header + "\n")

    # helper to process a single cast (CTD or manual)
    def process_cast_block(df_block=None, manual_meta=None):
        """
        Build a table of CTD or manual‐defined casts, interpolate the model
        (thetao & so) either at original pressures or a fine vertical grid,
        optionally smooth, and write every row out in ODV format.
        """
        # --- build the initial DataFrame for this cast ---
        if manual_meta:
            rows = []
            for p in manual_meta['PRESSURES']:
                row = {
                    'Cruise':       f"Model_{manual_meta['Cruise']}",
                    'Station':      manual_meta['Station'],
                    'Type':         'C',
                    'yyyy-mm-ddThh:mm:ss.sss': manual_meta['datetime'],
                    'Longitude [degrees_east]': manual_meta['Longitude [degrees_east]'],
                    'Latitude [degrees_north]': manual_meta['Latitude [degrees_north]'],
                    'LOCAL_CDI_ID': f"Model_{manual_meta['LOCAL_CDI_ID']}",
                    'EDMO_code':    '',
                    'Bot. Depth [m]': '',
                    'PRES':         p
                }
                # zero‐out any QV: columns
                for c in cols:
                    if c.startswith('QV:'):
                        row[c] = 0
                rows.append(row)
            df = pd.DataFrame(rows)
            df['datetime_parsed'] = pd.to_datetime(df['yyyy-mm-ddThh:mm:ss.sss'])
        else:
            df = df_block.copy()
            df['Cruise']       = df['Cruise'].apply(lambda x: f"Model_{x}")
            df['LOCAL_CDI_ID'] = df['LOCAL_CDI_ID'].apply(lambda x: f"Model_{x}")

        # --- validate time & location ---
        t0   = np.datetime64(df.loc[0, 'datetime_parsed'])
        lat0 = df.loc[0, 'Latitude [degrees_north]']
        lon0 = df.loc[0, 'Longitude [degrees_east]']
        if t0 < t_min or t0 > t_max:
            raise ValueError(f"{df.loc[0,'Station']}: date {t0} out of model time range")
        if not (lat_min <= lat0 <= lat_max and lon_min <= lon0 <= lon_max):
            raise ValueError(f"{df.loc[0,'Station']}: coords ({lat0},{lon0}) out of model domain")

        # --- find nearest model time & horizontal grid cell ---
        t_mod = ds_t['time'].sel(time=t0, method='nearest').values
        ilat  = np.searchsorted(lats, lat0)
        ilon  = np.searchsorted(lons, lon0)
        y0, y1 = lats[ilat-1], lats[ilat]
        x0, x1 = lons[ilon-1], lons[ilon]

        # --- 1) vertical up‐sample OR nearest‐depth fallback ---
        if apply_vertical_interp:
            p_min, p_max = df['PRES'].min(), df['PRES'].max()
            fine_p       = np.linspace(p_min, p_max, vertical_levels)
            da_t = ds_t['thetao'].sel(time=t_mod, method='nearest') \
                        .interp(latitude=lat0, longitude=lon0, depth=fine_p)
            da_s = ds_s['so']  .sel(time=t_mod, method='nearest') \
                        .interp(latitude=lat0, longitude=lon0, depth=fine_p)
            temps = da_t.values
            psals = da_s.values
            depths = fine_p
        else:
            temps, psals, depths = [], [], df['PRES'].values
            for _, row in df.iterrows():
                pres = row['PRES']
                wts  = bilinear_w(lon0, lat0, x0, x1, y0, y1)
                # sample four corners
                Tvals = {
                    'll': sample(ds_t,'thetao',t_mod,y0,x0,pres),
                    'lr': sample(ds_t,'thetao',t_mod,y0,x1,pres),
                    'ul': sample(ds_t,'thetao',t_mod,y1,x0,pres),
                    'ur': sample(ds_t,'thetao',t_mod,y1,x1,pres),
                }
                Svals = {
                    'll': sample(ds_s,'so',  t_mod,y0,x0,pres),
                    'lr': sample(ds_s,'so',  t_mod,y0,x1,pres),
                    'ul': sample(ds_s,'so',  t_mod,y1,x0,pres),
                    'ur': sample(ds_s,'so',  t_mod,y1,x1,pres),
                }
                temps.append(nan_interp(Tvals, wts))
                psals.append(nan_interp(Svals, wts))
            temps = np.array(temps)
            psals = np.array(psals)

        # --- 2) optional smoothing ---
        if apply_smoothing:
            temps = running_mean(temps, smoothing_window)
            psals = running_mean(psals, smoothing_window)

        # --- 3) write every (depth, temp, psal) row ---
        for p, Tm, Sm in zip(depths, temps, psals):
            # reuse first‐row metadata, override PRES & values
            meta = df.iloc[0].to_dict()
            meta['PRES'] = p
            out = []
            for k, col in enumerate(cols):
                if k == i_cruise:
                    out.append(meta['Cruise'])
                elif k == i_local_cdi:
                    out.append(meta['LOCAL_CDI_ID'])
                elif k == i_psal:
                    out.append(f"{Sm:.4f}")
                elif k == i_temp:
                    out.append(f"{Tm:.4f}")
                elif col == 'PRES':
                    out.append(str(p))
                else:
                    v = meta.get(col, '')
                    out.append('' if pd.isna(v) else str(v))
            outf.write("\t".join(out) + "\n")





    # 2) process CTD file casts
    if use_ctd:
        for idx in range(len(starts) - 1):
            block = ctd.iloc[starts[idx] : starts[idx + 1]].reset_index(drop=True)
            #process_cast_block(df_block=block)
            try:
                process_cast_block(df_block=block)
            except ValueError as e:
                skipped.append(str(e))
    # 3) process manual stations
    # Option A: hard‐coded manual stations
    if use_manual:
        for meta in manual_stations:
                        process_cast_block(manual_meta=meta)
    #Option B: read extra stations from a simple CSV

    if use_manual_csv:
        import csv
        df_pts = pd.read_csv(manual_csv_path, sep=',')
        for _, row in df_pts.iterrows():
              # parse the semi‐colon list of pressures
                    pressures = [float(x) for x in str(row['PRESSURES']).split(';') if x]
                    meta = {
                            'Station': row['Station'],
                            'Cruise': row['Cruise'],
                            'LOCAL_CDI_ID': row['LOCAL_CDI_ID'],
                            'datetime': row['datetime'],
                            'Latitude [degrees_north]': float(row['Latitude [degrees_north]']),
                            'Longitude [degrees_east]': float(row['Longitude [degrees_east]']),
                            'PRESSURES': pressures
                    }
                    process_cast_block(manual_meta=meta)

 #summary of skipped stations
if skipped:
    print("Skipped the following stations due to out-of-range:")
    for msg in skipped:
        print(" -", msg)

print("Wrote combined ODV file to:", output_file)
