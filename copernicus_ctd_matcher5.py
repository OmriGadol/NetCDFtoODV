import os
import re
import pandas as pd
import xarray as xr
import numpy as np
import gsw                                # <— TEOS-10 routines
import rasterio
import re
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
temp_nc     = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/2014/med-cmcc-tem-rean-d_1747652880573.nc'
sal_nc      = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/2014/med-cmcc-sal-rean-d_1747653098966.nc'
#output_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/Coper_model_only.txt'
output_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/ModelBasedProjected_30-10-14_Daily_00045.txt'
out_dir = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/'
slope_tif = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Batymetry/Israel_50m_slope.tif'
bathy_tif = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Batymetry/Israel_50m_scaled.tif'
manual_csv_path = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Batymetry/00045GEOM.csv'
# -------------------------------
# ====================================================
#                USER OPTIONS
# ====================================================
use_ctd    = False    # If True, pull stations from the CTD file
use_manual = False  # If True, append hard-codded stations from manual_stations list
use_manual_csv = True  # If True, read extra stations from CSV file
use_bathy = True     # If True, sample bottom depth from a GeoTIFF
use_slope       = True    # If True, sample seafloor slope at each station
apply_smoothing        = False  # If True, apply 1D running mean after (or without) vertical interp
compute_pycnocline     = True  # <— If True, locate depth of max dρ/dz
apply_vertical_interp  = True   # If True, upsample depths & linearly interpolate vertically
sharpen_pyc    = True       # If True, add extra depth points around the detected pycnocline
pyc_window     = 5.0        # ± meters around pycnocline to insert extra levels
pyc_npoints    = 7          # how many extra levels in that window
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
# -------------------------------

# -------------------------------
# Read ODV header & columns
# -------------------------------
with open(ctd_file, 'r') as f:
    lines = f.readlines()
hdr_idx    = next(i for i, L in enumerate(lines) if L.startswith("Cruise"))
header     = lines[:hdr_idx]
col_header = lines[hdr_idx].rstrip("\n") + "\tSlope [deg]"
cols       = col_header.split("\t")

i_cruise    = cols.index("Cruise")
i_local_cdi = cols.index("LOCAL_CDI_ID")
i_psal      = cols.index('PSALST01_UPPT')
i_temp      = cols.index("TEMPS901_UPAA")
i_slope     = cols.index("Slope [deg]")


# -------------------------------
# Load & clean CTD data
# -------------------------------
raw = pd.read_csv(ctd_file, sep="\t", skiprows=hdr_idx)
ctd = raw.ffill()
ctd["datetime_parsed"] = pd.to_datetime(ctd["yyyy-mm-ddThh:mm:ss.sss"])
starts = raw.index[raw["Type"] == "C"].tolist()
starts.append(len(ctd))

# -------------------------------
# Open model datasets & bathy
# -------------------------------
ds_t = xr.open_dataset(temp_nc)
ds_s = xr.open_dataset(sal_nc)
if use_bathy:
    bathy_src = rasterio.open(bathy_tif)
if use_slope:
    slope_src = rasterio.open(slope_tif)

# model ranges
t_min, t_max     = ds_t.time.min().values, ds_t.time.max().values
lat_min, lat_max = ds_t.latitude.min().values, ds_t.latitude.max().values
lon_min, lon_max = ds_t.longitude.min().values, ds_t.longitude.max().values
lats = ds_t.latitude.values
lons = ds_t.longitude.values

# helpers
def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='same')

def bilinear_w(x, y, x0, x1, y0, y1):
    dx, dy = x1-x0, y1-y0
    return {
        'll': (x1-x)*(y1-y)/(dx*dy),
        'lr': (x-x0)*(y1-y)/(dx*dy),
        'ul': (x1-x)*(y-y0)/(dx*dy),
        'ur': (x-x0)*(y-y0)/(dx*dy),
    }

def sample(ds, var, t, lat, lon, pres):
    return float(ds[var].sel(time=t, latitude=lat,
                             longitude=lon, depth=pres,
                             method='nearest').values)

def nan_interp(vals, wts):
    valid = {k: wts[k] for k,v in vals.items() if not np.isnan(vals[k])}
    if not valid: return np.nan
    W = sum(valid.values())
    return sum(vals[k]*valid[k] for k in valid)/W

# -------------------------------
# Process & write output
# -------------------------------
skipped = []
with open(output_file, "w") as outf:
    outf.writelines(header)
    outf.write(col_header+"\n")

    def process_cast_block(df_block=None, manual_meta=None):
        # build df for this cast
        if manual_meta:
            rows=[]
            for p in manual_meta['PRESSURES']:
                r = {
                    'Cruise': f"Model_{manual_meta['Cruise']}",
                    'Station': manual_meta['Station'],
                    'Type':'C',
                    'yyyy-mm-ddThh:mm:ss.sss': manual_meta['datetime'],
                    'Longitude [degrees_east]': manual_meta['Longitude [degrees_east]'],
                    'Latitude [degrees_north]': manual_meta['Latitude [degrees_north]'],
                    'LOCAL_CDI_ID': f"Model_{manual_meta['LOCAL_CDI_ID']}",
                    'EDMO_code':'','Bot. Depth [m]':'','PRES':p
                }
                for c in cols:
                    if c.startswith('QV:'): r[c]=0
                rows.append(r)
            df = pd.DataFrame(rows)
            df['datetime_parsed'] = pd.to_datetime(df['yyyy-mm-ddThh:mm:ss.sss'])
        else:
            df = df_block.copy()
            df['Cruise']       = df['Cruise'].apply(lambda x:f"Model_{x}")
            df['LOCAL_CDI_ID'] = df['LOCAL_CDI_ID'].apply(lambda x:f"Model_{x}")

        # assign bathy as bottom depth
        if use_bathy:
            lon0 = float(df.loc[0,'Longitude [degrees_east]'])
            lat0 = float(df.loc[0,'Latitude [degrees_north]'])
            rawd = bathy_src.sample([(lon0,lat0)]).__next__()[0]
            df['Bot. Depth [m]'] = abs(rawd)

        # --- NEW: sample seafloor slope at this lon/lat ---
        # assign slope if requested

        if use_slope:
            lon0 = float(df.loc[0, 'Longitude [degrees_east]'])
            lat0 = float(df.loc[0, 'Latitude [degrees_north]'])
            rawslope = slope_src.sample([(lon0, lat0)]).__next__()[0]
            df['Slope [deg]'] = float(rawslope)

        # validate
        t0 = np.datetime64(df.loc[0,'datetime_parsed'])
        lat0, lon0 = df.loc[0,'Latitude [degrees_north]'], df.loc[0,'Longitude [degrees_east]']
        if not (t_min<=t0<=t_max):
            skipped.append((df.loc[0,'Station'], 'time')); return
        if not (lat_min<=lat0<=lat_max and lon_min<=lon0<=lon_max):
            skipped.append((df.loc[0,'Station'], 'location')); return

        # nearest model time & grid
        t_mod = ds_t.time.sel(time=t0, method='nearest').values
        ilat  = np.searchsorted(lats, lat0)
        ilon  = np.searchsorted(lons, lon0)
        y0,y1 = lats[ilat-1], lats[ilat]
        x0,x1 = lons[ilon-1], lons[ilon]

        # 1) build temps, psals, depths
        if apply_vertical_interp:
            pmin,pmax = df['PRES'].min(), df['PRES'].max()
            fine_p  = np.linspace(pmin,pmax,vertical_levels)
            da_t = ds_t['thetao'].sel(time=t_mod, latitude=lat0, longitude=lon0, method='nearest')\
                     .interp(depth=fine_p)
            da_s = ds_s['so'].sel(time=t_mod, latitude=lat0, longitude=lon0, method='nearest')\
                     .interp(depth=fine_p)
            temps   = da_t.values; psals = da_s.values; depths = fine_p
        else:
            temps, psals, depths = [], [], df['PRES'].values
            for _,r in df.iterrows():
                p = r['PRES']
                w = bilinear_w(lon0,lat0,x0,x1,y0,y1)
                Tvals = {k: sample(ds_t,'thetao',t_mod,y0 if k in ['ll','lr'] else y1,
                                   x0 if k in ['ll','ul'] else x1, p)
                         for k in ['ll','lr','ul','ur']}
                Svals = {k: sample(ds_s,'so',  t_mod,y0 if k in ['ll','lr'] else y1,
                                   x0 if k in ['ll','ul'] else x1, p)
                         for k in ['ll','lr','ul','ur']}
                temps.append(nan_interp(Tvals,w))
                psals.append(nan_interp(Svals,w))
            temps = np.array(temps); psals = np.array(psals)

        original_depths = depths.copy()
        original_temps  = temps.copy()
        original_psals  = psals.copy()

        # 2) pycnocline detection
        if compute_pycnocline:
            SA = gsw.SA_from_SP(psals, depths, lon0, lat0)
            CT      = gsw.CT_from_t(SA, temps, depths)
            rho     = gsw.rho(SA, CT, depths)
            drho_dz = np.diff(rho) / np.diff(depths)
            idx     = np.nanargmax(np.abs(drho_dz))
            pyc     = 0.5*(depths[idx] + depths[idx+1])
            # clearly report which pycnocline depth we’ll use:
            station_id = df.loc[0, 'Station']
            print(f"→ Using pycnocline at {pyc:.1f} dbar for station {station_id}")

        # 3) sharpen around pycnocline
        if sharpen_pyc:
            dTdz = np.abs(np.gradient(original_temps, original_depths))
            idx  = np.nanargmax(dTdz)
            pd0  = original_depths[idx]
            extra = np.linspace(pd0-pyc_window, pd0+pyc_window, pyc_npoints)
            depths = np.unique(np.concatenate([original_depths, extra]))
            depths.sort()
            temps  = np.interp(depths, original_depths, original_temps)
            psals  = np.interp(depths, original_depths, original_psals)

        # 4) smoothing
        if apply_smoothing:
            temps = running_mean(temps, smoothing_window)
            psals = running_mean(psals, smoothing_window)

        # 5) write out: all original columns, but overwrite PSAL/TEMP/(optional)Slope/PRES
        for p, Tm, Sm in zip(depths, temps, psals):
            # grab the first-row metadata and override PRES and, if used, Slope
            meta = df.iloc[0].to_dict()
            meta['PRES'] = p
            if use_slope:
                meta['Slope [deg]'] = df.loc[0, 'Slope [deg]']

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
                elif use_slope and k == i_slope:
                    out.append(f"{meta['Slope [deg]']:.1f}")
                else:
                    val = meta.get(col, "")
                    out.append("" if pd.isna(val) else str(val))

            outf.write("\t".join(out) + "\n")


    # process CTD casts
    if use_ctd:
        for ix in range(len(starts)-1):
            blk = ctd.iloc[starts[ix]:starts[ix+1]].reset_index(drop=True)
            process_cast_block(df_block=blk)
    # process manual CSV

    if use_manual_csv:
        manual = pd.read_csv(manual_csv_path)
        for _, row in manual.iterrows():
          # allow either comma or semicolon separators
            pres_list = [float(x) for x in re.split(r'[;,]', str(row['PRESSURES'])) if x.strip()]
            meta = row.to_dict()
            meta['PRESSURES'] = pres_list
            process_cast_block(manual_meta=meta)
print("Done. Skipped:", skipped)
