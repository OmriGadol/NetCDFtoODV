import os
import re
import pandas as pd
import xarray as xr
import numpy as np
import gsw                                # TEOS-10 routines
import rasterio
from datetime import datetime

"""
===============================================================================
Script: generate_model_only_fullODV.py  (robust headers + CMEMS coord fix)

Description:
    Reads a CTD ODV text file (old or new header variants) and Copernicus
    model NetCDF outputs, interpolates temperature (thetao) and salinity (so)
    at each CTD station's location and depths, and writes a single combined
    ODV-compatible output file containing only the model data for each station.

    - Supports old ODV-style columns:
        'yyyy-mm-ddThh:mm:ss.sss', 'Longitude [degrees_east]',
        'Latitude [degrees_north]', 'PRES', 'TEMPS901_UPAA', 'PSALST01_UPPT',
        optional 'LOCAL_CDI_ID', etc.
    - Supports newer CTD columns:
        'mon/day/yr', 'hh:mm', 'Lon (?E)', 'Lat (?N)', 'Pressure [db]',
        'Temperature [C]', 'Salinity [psu]'.

Usage:
    python generate_model_only_fullODV.py

Configuration:
    - Toggle `use_ctd`, `use_manual`, `use_manual_csv` in USER OPTIONS.
    - Define file paths and optional manual_stations/CSV as needed.

Notes:
    - Output preserves the original header and column order from the CTD file.
    - Adds 'Slope [deg]' only when use_slope=True.
    - Auto-standardizes CMEMS coords (lat/lon/depth/time) and uses xarray interp.
===============================================================================
"""

# -------------------------------
# 1. File paths
# -------------------------------
ctd_file    = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/HaiSec35_ODV.txt'
temp_nc     = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/2016/med-cmcc-tem-rean-d_1755125149387.nc'
sal_nc      = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/2016/med-cmcc-sal-rean-d_1755125057633.nc'
output_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/ModelBasedHaiSec35_ODV.txt'
out_dir     = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/'
slope_tif   = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Batymetry/Israel_50m_slope.tif'
bathy_tif   = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Batymetry/Israel_50m_scaled.tif'
manual_csv_path = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Batymetry/00045GEOM.csv'

# ====================================================
#                USER OPTIONS
# ====================================================
use_ctd               = True     # If True, pull stations from the CTD file
use_manual            = False    # If True, append hard-coded stations list (see template below)
use_manual_csv        = False    # If True, read extra stations from CSV file
use_bathy             = False    # If True, sample bottom depth from GeoTIFF
use_slope             = False    # If True, sample seafloor slope at each station
apply_smoothing       = False    # If True, apply 1D running mean after sampling
compute_pycnocline    = False    # If True, locate depth of max dρ/dz
apply_vertical_interp = False    # If True, upsample depths & linearly interpolate vertically
sharpen_pyc           = False    # If True, add extra depth points around the detected pycnocline
pyc_window            = 5.0      # ± meters around pycnocline to insert extra levels
pyc_npoints           = 7        # how many extra levels in that window
vertical_levels       = 50       # Number of fine depth levels when vertical_interp is True
smoothing_window      = 5        # window size (number of depth points) for running mean

# -------------------------------
# Helper: resolve column names across header variants
# -------------------------------
def first_present(cols, candidates, default=None):
    for c in candidates:
        if c in cols:
            return c
    return default

def parse_datetime_cols(df, cols):
    """
    Returns a pd.Series of parsed datetimes from either:
      - single ISO column 'yyyy-mm-ddThh:mm:ss.sss'
      - or separate 'mon/day/yr' + 'hh:mm' (or hh:mm:ss)
    """
    iso_col = first_present(cols, ['yyyy-mm-ddThh:mm:ss.sss', 'DateTime_ISO'])
    if iso_col:
        return pd.to_datetime(df[iso_col], errors='coerce')

    date_col = first_present(cols, ['mon/day/yr', 'mm/dd/yy', 'mm/dd/yyyy'])
    time_col = first_present(cols, ['hh:mm:ss', 'hh:mm', 'time'])
    if date_col and time_col:
        def _parse(r):
            d = str(r[date_col]).strip()
            t = str(r[time_col]).strip()
            # Try multiple date formats robustly
            for fmt in ('%m/%d/%Y', '%m/%d/%y'):
                for tfmt in ('%H:%M:%S', '%H:%M'):
                    try:
                        return pd.to_datetime(f'{d} {t}', format=f'{fmt} {tfmt}', errors='raise')
                    except Exception:
                        pass
            # Fallback: pandas guess
            try:
                return pd.to_datetime(f'{d} {t}', errors='coerce')
            except Exception:
                return pd.NaT
        return df.apply(_parse, axis=1)

    # Last fallback: try any 'Date*' + 'Time*'
    date_like = [c for c in cols if c.lower().startswith('date')]
    time_like = [c for c in cols if c.lower().startswith('time')]
    if date_like and time_like:
        return pd.to_datetime(df[date_like[0]] + ' ' + df[time_like[0]], errors='coerce')
    return pd.to_datetime(pd.Series([None]*len(df)))  # all NaT

def running_mean(x, N):
    if len(x) == 0: return x
    N = int(max(1, N))
    return np.convolve(x, np.ones(N)/float(N), mode='same')

# -------------------------------
# OPTIONAL: Define manual stations if use_manual=True
# -------------------------------
# Example template:
# manual_stations = [
#     {
#         'Cruise': 'CustomCast1',
#         'Station': 'MAN1',
#         'LOCAL_CDI_ID': 'Manual001',
#         'datetime': '2014-10-23T07:00:00.000',
#         'Latitude [degrees_north]': 32.65,
#         'Longitude [degrees_east]': 34.80,
#         'PRESSURES': [1, 10, 20, 50, 100, 120, 150, 200, 220]
#     },
# ]
manual_stations = []

# -------------------------------
# Read ODV header & columns
# -------------------------------
with open(ctd_file, 'r') as f:
    lines = f.readlines()

hdr_idx_candidates = [i for i, L in enumerate(lines) if L.strip().startswith("Cruise")]
if not hdr_idx_candidates:
    raise RuntimeError("Could not find a header line starting with 'Cruise'.")
hdr_idx = hdr_idx_candidates[0]

# Base column header from file
base_col_header = lines[hdr_idx].rstrip("\n")
cols = base_col_header.split("\t")

# Decide if/when to append 'Slope [deg]' column
col_header = base_col_header
if use_slope and ('Slope [deg]' not in cols):
    col_header += "\tSlope [deg]"
    cols = col_header.split("\t")

# Identify canonical column names (old + new)
COL_CRUISE   = first_present(cols, ['Cruise'])
COL_STATION  = first_present(cols, ['Station'])
COL_TYPE     = first_present(cols, ['Type'])
COL_LON      = first_present(cols, ['Longitude [degrees_east]', 'Lon (?E)', 'Longitude [deg E]', 'Longitude'])
COL_LAT      = first_present(cols, ['Latitude [degrees_north]', 'Lat (?N)', 'Latitude [deg N]', 'Latitude'])
COL_PRES     = first_present(cols, ['PRES', 'Pressure [db]', 'Pressure [dbar]', 'Pressure'])
COL_TEMP     = first_present(cols, ['TEMPS901_UPAA', 'Temperature [C]', 'TEMP', 'Temperature'])
COL_SAL      = first_present(cols, ['PSALST01_UPPT', 'Salinity [psu]', 'PSAL', 'Salinity'])
COL_LOCALCDI = first_present(cols, ['LOCAL_CDI_ID'])  # may be None

# -------------------------------
# Load & clean CTD data
# -------------------------------
raw = pd.read_csv(ctd_file, sep="\t", skiprows=hdr_idx)
ctd = raw.ffill()

# Build parsed datetime column from either ISO or date+time
ctd["datetime_parsed"] = parse_datetime_cols(ctd, list(raw.columns))

# Station block starts (prefer Type=='C'; fallback to every Station change)
if COL_TYPE in raw.columns and 'C' in set(raw[COL_TYPE].astype(str)):
    starts = raw.index[raw[COL_TYPE] == "C"].tolist()
else:
    # Fallback: treat consecutive rows with same station as a block
    stn_col = COL_STATION if COL_STATION in raw.columns else raw.columns[0]
    starts = [0] + (raw[stn_col].shift() != raw[stn_col]).to_numpy().nonzero()[0].tolist()
    starts = sorted(set(starts))
starts.append(len(ctd))

# -------------------------------
# Open model datasets & make coords consistent
# -------------------------------
ds_t = xr.open_dataset(temp_nc)
ds_s = xr.open_dataset(sal_nc)

def standardize_cmems(ds):
    # Find actual coord names
    lat_name  = next((c for c in ['latitude','lat','nav_lat','y'] if c in ds.coords or c in ds.dims), None)
    lon_name  = next((c for c in ['longitude','lon','nav_lon','x'] if c in ds.coords or c in ds.dims), None)
    dep_name  = next((c for c in ['depth','lev','deptht','z'] if c in ds.coords or c in ds.dims), None)
    time_name = next((c for c in ['time','t'] if c in ds.coords or c in ds.dims), None)
    rename_map = {}
    if lat_name  and lat_name  != 'latitude':  rename_map[lat_name]  = 'latitude'
    if lon_name  and lon_name  != 'longitude': rename_map[lon_name]  = 'longitude'
    if dep_name  and dep_name  != 'depth':     rename_map[dep_name]  = 'depth'
    if time_name and time_name != 'time':      rename_map[time_name] = 'time'
    if rename_map:
        ds = ds.rename(rename_map)

    # Ensure they are coords (not just dims)
    for c in ['time','latitude','longitude','depth']:
        if c in ds.dims and c not in ds.coords:
            ds = ds.assign_coords({c: ds[c]})

    # Sort lat/lon ascending so search/interp behave predictably
    if 'latitude' in ds.coords:
        if np.any(np.diff(ds['latitude'].values) < 0):
            ds = ds.sortby('latitude')
    if 'longitude' in ds.coords:
        if np.any(np.diff(ds['longitude'].values) < 0):
            ds = ds.sortby('longitude')

    return ds

ds_t = standardize_cmems(ds_t)
ds_s = standardize_cmems(ds_s)

# Try to detect variable names if different from 'thetao' / 'so'
def detect_var(ds, preferred, fallbacks=()):
    if preferred in ds.data_vars:
        return preferred
    for cand in fallbacks:
        if cand in ds.data_vars:
            return cand
    # Guess by CF standard_name / long_name
    for v in ds.data_vars:
        da = ds[v]
        std = da.attrs.get('standard_name','').lower()
        lon = da.attrs.get('long_name','').lower()
        if 'potential_temperature' in std or 'potential temperature' in lon:
            return v
        if 'sea_water_salinity' in std or 'salinity' in lon:
            if preferred == 'so':
                return v
    return preferred  # fallback; will error later if missing

var_temp = detect_var(ds_t, 'thetao', ('temp','temperature'))
var_sal  = detect_var(ds_s, 'so',     ('salinity','psal','salt'))

# Model ranges
t_min_t, t_max_t = ds_t.time.min().values, ds_t.time.max().values
t_min_s, t_max_s = ds_s.time.min().values, ds_s.time.max().values
lat_min = max(ds_t.latitude.min().item(), ds_s.latitude.min().item())
lat_max = min(ds_t.latitude.max().item(), ds_s.latitude.max().item())
lon_min = max(ds_t.longitude.min().item(), ds_s.longitude.min().item())
lon_max = min(ds_t.longitude.max().item(), ds_s.longitude.max().item())

# Open rasters if requested
if use_bathy:
    bathy_src = rasterio.open(bathy_tif)
if use_slope:
    slope_src = rasterio.open(slope_tif)

# -------------------------------
# Process & write output
# -------------------------------
skipped = []
with open(output_file, "w") as outf:
    # Write original pre-header (if any) + adjusted header line
    header = lines[:hdr_idx]
    outf.writelines(header)
    outf.write(col_header + "\n")

    def process_cast_block(df_block=None, manual_meta=None):
        # Build df for this cast
        if manual_meta:
            rows=[]
            for p in manual_meta['PRESSURES']:
                r = {
                    COL_CRUISE: f"Model_{manual_meta.get(COL_CRUISE, manual_meta.get('Cruise','CustomCast'))}",
                    'Station': manual_meta.get('Station','Manual'),
                    'Type':'C',
                    'yyyy-mm-ddThh:mm:ss.sss': manual_meta.get('datetime', None),
                    COL_LON: manual_meta.get(COL_LON, manual_meta.get('Longitude [degrees_east]', None)),
                    COL_LAT: manual_meta.get(COL_LAT, manual_meta.get('Latitude [degrees_north]', None)),
                    'Bot. Depth [m]': manual_meta.get('Bot. Depth [m]', ''),
                }
                if COL_PRES:
                    r[COL_PRES] = p
                if COL_LOCALCDI:
                    r[COL_LOCALCDI] = f"Model_{manual_meta.get(COL_LOCALCDI, 'Manual001')}"
                rows.append(r)
            df = pd.DataFrame(rows)
            # datetime_parsed
            if 'yyyy-mm-ddThh:mm:ss.sss' in df.columns:
                df['datetime_parsed'] = pd.to_datetime(df['yyyy-mm-ddThh:mm:ss.sss'], errors='coerce')
            else:
                df['datetime_parsed'] = pd.to_datetime(manual_meta.get('datetime', None))
            # ensure lon/lat numeric
            if COL_LON in df.columns: df[COL_LON] = pd.to_numeric(df[COL_LON], errors='coerce')
            if COL_LAT in df.columns: df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors='coerce')
        else:
            df = df_block.copy()
            # Prefix Cruise/LOCAL_CDI_ID with 'Model_' if present
            if COL_CRUISE in df.columns:
                df[COL_CRUISE] = df[COL_CRUISE].astype(str).apply(lambda x: f"Model_{x}")
            if COL_LOCALCDI and (COL_LOCALCDI in df.columns):
                df[COL_LOCALCDI] = df[COL_LOCALCDI].astype(str).apply(lambda x: f"Model_{x}")

            # If no unified pressure column exists, create it from the best candidate
            if COL_PRES not in df.columns:
                pres_col = first_present(df.columns, ['PRES', 'Pressure [db]', 'Pressure [dbar]', 'Pressure'])
                if pres_col:
                    df[COL_PRES] = pd.to_numeric(df[pres_col], errors='coerce')

        # Assign bathy as bottom depth if requested
        if use_bathy:
            lon0 = float(df.loc[0, COL_LON])
            lat0 = float(df.loc[0, COL_LAT])
            rawd = bathy_src.sample([(lon0,lat0)]).__next__()[0]
            df['Bot. Depth [m]'] = abs(rawd)

        # Assign slope if requested
        if use_slope:
            lon0 = float(df.loc[0, COL_LON])
            lat0 = float(df.loc[0, COL_LAT])
            rawslope = slope_src.sample([(lon0, lat0)]).__next__()[0]
            df['Slope [deg]'] = float(rawslope)

        # Validate time & location against BOTH datasets
        t0 = np.datetime64(df.loc[0,'datetime_parsed'])
        lat0, lon0 = float(df.loc[0, COL_LAT]), float(df.loc[0, COL_LON])
        in_time = (t_min_t <= t0 <= t_max_t) and (t_min_s <= t0 <= t_max_s)
        in_area = (lat_min <= lat0 <= lat_max) and (lon_min <= lon0 <= lon_max)
        if not in_time:
            skipped.append((df.loc[0, COL_STATION] if COL_STATION in df.columns else 'UNK', 'time')); return
        if not in_area:
            skipped.append((df.loc[0, COL_STATION] if COL_STATION in df.columns else 'UNK', 'location')); return

        # pick nearest time independently for T and S (calendars/ranges can differ)
        t_mod_t = ds_t['time'].sel(time=t0, method='nearest')
        t_mod_s = ds_s['time'].sel(time=t0, method='nearest')

        # Build arrays at requested pressures
        pres_vals = pd.to_numeric(df[COL_PRES], errors='coerce').values

        if apply_vertical_interp:
            pmin, pmax = np.nanmin(pres_vals), np.nanmax(pres_vals)
            fine_p  = np.linspace(pmin, pmax, vertical_levels)

            da_t = (ds_t[var_temp]
                    .sel(time=t_mod_t)
                    .interp(latitude=float(lat0), longitude=float(lon0))
                    .interp(depth=fine_p))
            da_s = (ds_s[var_sal]
                    .sel(time=t_mod_s)
                    .interp(latitude=float(lat0), longitude=float(lon0))
                    .interp(depth=fine_p))

            depths = fine_p
            temps  = da_t.values
            psals  = da_s.values
        else:
            depths, temps, psals = [], [], []
            for p in pres_vals:
                if np.isnan(p):
                    depths.append(np.nan); temps.append(np.nan); psals.append(np.nan); continue
                Tp = (ds_t[var_temp]
                      .sel(time=t_mod_t)
                      .interp(latitude=float(lat0), longitude=float(lon0))
                      .interp(depth=float(p))
                      .values)
                Sp = (ds_s[var_sal]
                      .sel(time=t_mod_s)
                      .interp(latitude=float(lat0), longitude=float(lon0))
                      .interp(depth=float(p))
                      .values)
                depths.append(p)
                temps.append(np.array(Tp).item() if np.size(Tp)==1 else np.nan)
                psals.append(np.array(Sp).item() if np.size(Sp)==1 else np.nan)

            depths = np.array(depths)
            temps  = np.array(temps)
            psals  = np.array(psals)

        # Drop rows where both are NaN
        good = (~np.isnan(temps)) | (~np.isnan(psals))
        depths = np.array(depths)[good]
        temps  = np.array(temps)[good]
        psals  = np.array(psals)[good]

        original_depths = depths.copy()
        original_temps  = temps.copy()
        original_psals  = psals.copy()

        # Pycnocline (optional)
        if compute_pycnocline and (len(original_depths) > 2):
            try:
                SA  = gsw.SA_from_SP(original_psals, original_depths, lon0, lat0)
                CT  = gsw.CT_from_t(SA, original_temps, original_depths)
                rho = gsw.rho(SA, CT, original_depths)
                drho_dz = np.diff(rho) / np.diff(original_depths)
                idx     = np.nanargmax(np.abs(drho_dz))
                pyc     = 0.5*(original_depths[idx] + original_depths[idx+1])
                station_id = df.loc[0, COL_STATION] if COL_STATION in df.columns else 'UNK'
                print(f"→ Using pycnocline at {pyc:.1f} dbar for station {station_id}")
            except Exception:
                pass

        # Sharpen around pycnocline (optional)
        if sharpen_pyc and (len(original_depths) > 3):
            dTdz = np.abs(np.gradient(original_temps, original_depths))
            idx  = np.nanargmax(dTdz)
            pd0  = original_depths[idx]
            extra = np.linspace(pd0-pyc_window, pd0+pyc_window, pyc_npoints)
            depths = np.unique(np.concatenate([original_depths, extra]))
            depths.sort()
            temps  = np.interp(depths, original_depths, original_temps)
            psals  = np.interp(depths, original_depths, original_psals)

        # Smoothing (optional)
        if apply_smoothing:
            temps = running_mean(temps, smoothing_window)
            psals = running_mean(psals, smoothing_window)

        # -------- Write out one row per sampled depth --------
        out_cols = cols[:]  # preserve original column order
        if use_slope and ('Slope [deg]' not in out_cols): out_cols.append('Slope [deg]')
        if COL_TEMP and (COL_TEMP not in out_cols): out_cols.append(COL_TEMP)
        if COL_SAL  and (COL_SAL  not in out_cols):  out_cols.append(COL_SAL)
        if COL_PRES and (COL_PRES not in out_cols): out_cols.append(COL_PRES)

        for p, Tm, Sm in zip(depths, temps, psals):
            meta = df.iloc[0].to_dict()

            # Ensure mandatory fields exist for output
            if COL_PRES: meta[COL_PRES] = p
            if use_slope:
                meta['Slope [deg]'] = df.get('Slope [deg]', np.nan)

            # Ensure Cruise/LOCAL_CDI_ID are prefixed where present
            if COL_CRUISE in meta and isinstance(meta[COL_CRUISE], str) and not meta[COL_CRUISE].startswith('Model_'):
                meta[COL_CRUISE] = f"Model_{meta[COL_CRUISE]}"
            if COL_LOCALCDI and (COL_LOCALCDI in meta) and isinstance(meta[COL_LOCALCDI], str) and not meta[COL_LOCALCDI].startswith('Model_'):
                meta[COL_LOCALCDI] = f"Model_{meta[COL_LOCALCDI]}"

            # Place model values into appropriate columns (or create them)
            if COL_TEMP: meta[COL_TEMP] = f"{Tm:.4f}" if not np.isnan(Tm) else ""
            if COL_SAL:  meta[COL_SAL]  = f"{Sm:.4f}" if not np.isnan(Sm) else ""

            # Build output row in out_cols order
            out = []
            for col in out_cols:
                val = meta.get(col, "")
                if pd.isna(val): val = ""
                out.append(str(val))
            outf.write("\t".join(out) + "\n")

    # ---------- Process CTD casts ----------
    if use_ctd:
        for ix in range(len(starts)-1):
            blk = ctd.iloc[starts[ix]:starts[ix+1]].reset_index(drop=True)
            # Make sure lon/lat numeric
            if COL_LON in blk.columns: blk[COL_LON] = pd.to_numeric(blk[COL_LON], errors='coerce')
            if COL_LAT in blk.columns: blk[COL_LAT] = pd.to_numeric(blk[COL_LAT], errors='coerce')
            process_cast_block(df_block=blk)

    # ---------- Process manual stations ----------
    if use_manual and manual_stations:
        for meta in manual_stations:
            process_cast_block(manual_meta=meta)

    # ---------- Process manual CSV ----------
    if use_manual_csv:
        manual = pd.read_csv(manual_csv_path)
        for _, row in manual.iterrows():
            pres_list = [float(x) for x in re.split(r'[;,]', str(row.get('PRESSURES',''))) if x.strip()]
            meta = row.to_dict()
            meta['PRESSURES'] = pres_list
            process_cast_block(manual_meta=meta)

print("Done. Skipped:", skipped)
