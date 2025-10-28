

#!/usr/bin/env python3
"""
Merge SeaDataNet SADCP 'trajectoryProfile' NetCDFs into ONE ODV Spreadsheet (.txt)
— with quality fields — and WITHOUT degrading resolution:
• Each ensemble (each TIME step) gets its own Station ID => no coalescing.

Output (tab-delimited ODV Generic Spreadsheet):
Cruise | Station | Type | yyyy-mm-ddThh:mm:ss.sss | Longitude [degrees_east] | Latitude [degrees_north] |
Bot. Depth [m] | PROFZ [meters] |
UCUR | VCUR | WCUR | SPEED | DIRCUR | ECUR_ERROR |
PGOOD | CORR | NB_PINGS_PER_ENS |
USHIP | VSHIP | HEADING | PITCH | ROLL |
TIME_SEADATANET_QC | POSITION_SEADATANET_QC | PROFZ_SEADATANET_QC |
UCUR_SEADATANET_QC | VCUR_SEADATANET_QC | WCUR_SEADATANET_QC |
USHIP_SEADATANET_QC | VSHIP_SEADATANET_QC | HEADING_SEADATANET_QC |
PITCH_SEADATANET_QC | ROLL_SEADATANET_QC | BOTTOM_DEPTH_SEADATANET_QC | BATHY_SEADATANET_QC |
Source_File | LOCAL_CDI_ID
"""

# ------------------- SET THESE -------------------
INPUT_DIR = "//Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Cadiz/CurrentMeter/order_74890_unrestricted/nc/GOOD"  # <- change me
OUTPUT_PATH = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Cadiz/CurrentMeter/order_74890_unrestricted/nc/GOOD/ADCP_GOOD_merged.txt"
# -------------------------------------------------------

import os, glob, warnings
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import num2date

DEFAULT_CRUISE = "SADCP_CRUISE"
DEFAULT_TYPE   = "C"  # ODV continuous section

# SeaDataNet variable names
N = dict(
    TIME="TIME",
    LON="LONGITUDE",
    LAT="LATITUDE",
    PROFZ="PROFZ",
    U="UCUR",
    V="VCUR",
    W="WCUR",
    ECUR_ERR="ECUR",  # velocity error (NOT speed)
    PGOOD="PGOOD",
    CORR="CORR",
    NPINGS="NB_PINGS_PER_ENS",
    USHIP="USHIP",
    VSHIP="VSHIP",
    HEADING="HEADING",
    PITCH="PITCH",
    ROLL="ROLL",
    BOT="BOTTOM_DEPTH",     # station bottom (often missing)
    BATHY="BATHY",          # bathymetry along track
    CRUISE="SDN_CRUISE",
    STATION="SDN_STATION",
    CDI="SDN_LOCAL_CDI_ID",
    # QC flags
    Q_TIME="TIME_SEADATANET_QC",
    Q_POS="POSITION_SEADATANET_QC",
    Q_PROFZ="PROFZ_SEADATANET_QC",
    Q_U="UCUR_SEADATANET_QC",
    Q_V="VCUR_SEADATANET_QC",
    Q_W="WCUR_SEADATANET_QC",
    Q_USHIP="USHIP_SEADATANET_QC",
    Q_VSHIP="VSHIP_SEADATANET_QC",
    Q_HEAD="HEADING_SEADATANET_QC",
    Q_PITCH="PITCH_SEADATANET_QC",
    Q_ROLL="ROLL_SEADATANET_QC",
    Q_BOT="BOTTOM_DEPTH_SEADATANET_QC",
    Q_BATHY="BATHY_SEADATANET_QC",
)

def get(ds, name):
    return ds[name] if (name and name in ds.variables) else None

def decode_time_to_np_datetime64(time_da: xr.DataArray) -> np.ndarray:
    """Decode CF time to datetime64[ns]; handles Julian refs (CFYear0 warnings are OK)."""
    vals = np.array(time_da.squeeze().values, dtype="float64")
    units = str(time_da.attrs.get("units", ""))
    if units:
        try:
            cft = num2date(vals, units=units)  # may yield cftime objects
            ts = pd.to_datetime([str(t) for t in cft], errors="coerce")
            return ts.values.astype("datetime64[ns]")
        except Exception:
            pass
    # fallback (assume seconds since epoch)
    return (vals * 1e9).astype("datetime64[ns]")

def ensure_2d_from_1d(vec_1d: np.ndarray, tgt_shape_2d: tuple) -> np.ndarray:
    """Repeat a 1D (MAXT,) array to match (MAXT, MAXZ)."""
    vec_1d = np.array(vec_1d, dtype="float64").reshape(-1)
    MAXT, MAXZ = tgt_shape_2d
    return np.repeat(vec_1d[:, None], MAXZ, axis=1)

def read_scalar(var, default=""):
    try:
        if var is None: return default
        arr = np.array(var.values)
        if arr.size == 1:
            return str(arr.item())
    except Exception:
        pass
    return default

def _to_1d_per_time(arr_like, maxt):
    """Coerce a var (None/scalar/1D/2D) to 1D (MAXT,) per-time series."""
    if arr_like is None:
        return np.full(maxt, np.nan, dtype="float64")
    arr = np.array(arr_like.squeeze(), dtype="float64")
    if arr.ndim == 0:
        return np.full(maxt, arr.item(), dtype="float64")
    if arr.ndim == 1:
        return arr
    # If 2D (MAXT, MAXZ): reduce along depth (use nanmax so a single valid value wins)
    return np.nanmax(arr, axis=1)

def process_file(fp: str) -> pd.DataFrame:
    ds = xr.open_dataset(fp, decode_times=False)

    # Core vars
    TIME   = get(ds, N["TIME"])
    LON    = get(ds, N["LON"])
    LAT    = get(ds, N["LAT"])
    PROFZ  = get(ds, N["PROFZ"])
    U      = get(ds, N["U"])
    V      = get(ds, N["V"])
    W      = get(ds, N["W"])
    ECERR  = get(ds, N["ECUR_ERR"])
    PGOOD  = get(ds, N["PGOOD"])
    CORR   = get(ds, N["CORR"])
    NPING  = get(ds, N["NPINGS"])

    USHIP  = get(ds, N["USHIP"])
    VSHIP  = get(ds, N["VSHIP"])
    HEADING= get(ds, N["HEADING"])
    PITCH  = get(ds, N["PITCH"])
    ROLL   = get(ds, N["ROLL"])
    BOT    = get(ds, N["BOT"])
    BATHY  = get(ds, N["BATHY"])

    # QC flags (optional)
    Q_TIME  = get(ds, N["Q_TIME"]);   Q_POS   = get(ds, N["Q_POS"])
    Q_PROFZ = get(ds, N["Q_PROFZ"]);  Q_U     = get(ds, N["Q_U"])
    Q_V     = get(ds, N["Q_V"]);      Q_W     = get(ds, N["Q_W"])
    Q_USHIP = get(ds, N["Q_USHIP"]);  Q_VSHIP = get(ds, N["Q_VSHIP"])
    Q_HEAD  = get(ds, N["Q_HEAD"]);   Q_PITCH = get(ds, N["Q_PITCH"])
    Q_ROLL  = get(ds, N["Q_ROLL"]);   Q_BOT   = get(ds, N["Q_BOT"])
    Q_BATHY = get(ds, N["Q_BATHY"])

    # IDs / metadata
    CRUISE = get(ds, N["CRUISE"]); STN = get(ds, N["STATION"]); CDI = get(ds, N["CDI"])
    cruise   = read_scalar(CRUISE, DEFAULT_CRUISE)
    file_stn = read_scalar(STN, os.path.splitext(os.path.basename(fp))[0])
    cdi      = read_scalar(CDI, "")

    # Guard
    if TIME is None or PROFZ is None or U is None or V is None:
        ds.close()
        raise ValueError("Missing required variables: TIME, PROFZ, UCUR, VCUR")

    # Numpy arrays
    t_np   = decode_time_to_np_datetime64(TIME)                      # (MAXT,)
    z_np   = np.array(PROFZ.squeeze().values, dtype="float64")       # (MAXT, MAXZ)
    u_np   = np.array(U.squeeze().values, dtype="float64")
    v_np   = np.array(V.squeeze().values, dtype="float64")
    w_np   = (np.array(W.squeeze().values, dtype="float64")
              if W is not None else np.full_like(z_np, np.nan))
    e_np   = (np.array(ECERR.squeeze().values, dtype="float64")
              if ECERR is not None else np.full_like(z_np, np.nan))
    pg_np  = (np.array(PGOOD.squeeze().values, dtype="float64")
              if PGOOD is not None else np.full_like(z_np, np.nan))
    co_np  = (np.array(CORR.squeeze().values,  dtype="float64")
              if CORR  is not None else np.full_like(z_np, np.nan))

    MAXT, MAXZ = z_np.shape

    lon_1d = _to_1d_per_time(LON,   MAXT)
    lat_1d = _to_1d_per_time(LAT,   MAXT)
    npi_1d = _to_1d_per_time(NPING, MAXT)

    us_1d  = _to_1d_per_time(USHIP,  MAXT)
    vs_1d  = _to_1d_per_time(VSHIP,  MAXT)
    hd_1d  = _to_1d_per_time(HEADING,MAXT)
    pi_1d  = _to_1d_per_time(PITCH,  MAXT)
    ro_1d  = _to_1d_per_time(ROLL,   MAXT)

    # ---- Bottom: prefer BOTTOM_DEPTH where finite; else use BATHY; ensure positive ----
    bot_from_bot  = _to_1d_per_time(BOT,   MAXT)   # may be NaN
    bot_from_bty  = _to_1d_per_time(BATHY, MAXT)   # may be NaN
    # If BOT has NaNs, fill them from BATHY
    bot_1d = np.where(np.isfinite(bot_from_bot), bot_from_bot, bot_from_bty)
    # Make positive meters (ODV expects positive)
    with np.errstate(invalid="ignore"):
        bot_1d = np.abs(bot_1d)

    # Broadcast 1D -> 2D
    T2   = np.repeat(t_np.reshape(MAXT, 1), MAXZ, axis=1)
    LON2 = ensure_2d_from_1d(lon_1d, (MAXT, MAXZ))
    LAT2 = ensure_2d_from_1d(lat_1d, (MAXT, MAXZ))
    BOT2 = ensure_2d_from_1d(bot_1d, (MAXT, MAXZ))
    US2  = ensure_2d_from_1d(us_1d,  (MAXT, MAXZ))
    VS2  = ensure_2d_from_1d(vs_1d,  (MAXT, MAXZ))
    HD2  = ensure_2d_from_1d(hd_1d,  (MAXT, MAXZ))
    PI2  = ensure_2d_from_1d(pi_1d,  (MAXT, MAXZ))
    RO2  = ensure_2d_from_1d(ro_1d,  (MAXT, MAXZ))
    NP2  = ensure_2d_from_1d(npi_1d, (MAXT, MAXZ))

    # Station-per-ensemble to keep full temporal resolution
    t_iso_compact = pd.to_datetime(t_np).strftime("%Y%m%dT%H%M%S")
    station_per_time = np.array([f"{cruise}_{s}" for s in t_iso_compact])
    STATION2 = np.repeat(station_per_time[:, None], MAXZ, axis=1)

    # Derived
    with np.errstate(invalid="ignore"):
        SPEED  = np.sqrt(u_np**2 + v_np**2)
        DIRCUR = (np.degrees(np.arctan2(u_np, v_np)) + 360.0) % 360.0

    # QC -> numpy (broadcast 1D to 2D where needed)
    def to_np_or_nan(var, like2d=None):
        if var is None:
            return (np.full(like2d.shape, np.nan) if like2d is not None else None)
        arr = np.array(var.squeeze().values)
        if like2d is not None and arr.ndim == 1:
            return ensure_2d_from_1d(arr, like2d.shape)
        return arr

    q_time  = to_np_or_nan(Q_TIME)
    q_pos   = to_np_or_nan(Q_POS)
    q_profz = to_np_or_nan(Q_PROFZ, like2d=z_np)
    q_u     = to_np_or_nan(Q_U,     like2d=z_np)
    q_v     = to_np_or_nan(Q_V,     like2d=z_np)
    q_w     = to_np_or_nan(Q_W,     like2d=z_np)
    q_ush   = to_np_or_nan(Q_USHIP)
    q_vsh   = to_np_or_nan(Q_VSHIP)
    q_head  = to_np_or_nan(Q_HEAD)
    q_pitch = to_np_or_nan(Q_PITCH)
    q_roll  = to_np_or_nan(Q_ROLL)
    q_bot   = to_np_or_nan(Q_BOT)
    q_bathy = to_np_or_nan(Q_BATHY)

    def as1(x): return np.array(x).ravel()

    df = pd.DataFrame({
        "Cruise": cruise,
        "Station": as1(STATION2),
        "Type": DEFAULT_TYPE,
        "yyyy-mm-ddThh:mm:ss.sss": [pd.to_datetime(x).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                                    if not np.isnat(x) else "" for x in as1(T2)],
        "Longitude [degrees_east]": as1(LON2),
        "Latitude [degrees_north]": as1(LAT2),
        "Bot. Depth [m]": as1(BOT2),                     # <- now filled from BOT or BATHY (positive)
        "PROFZ [meters]": as1(z_np),
        "UCUR [meter/second]": as1(u_np),
        "VCUR [meter/second]": as1(v_np),
        "WCUR [meter/second]": as1(w_np),
        "SPEED [meter/second]": as1(SPEED),
        "DIRCUR [degrees]": as1(DIRCUR),
        "ECUR_ERROR [meter/second]": as1(e_np),
        "PGOOD [percent]": as1(pg_np),
        "CORR [percent]":  as1(co_np),
        "NB_PINGS_PER_ENS [dimless]": as1(NP2),
        "USHIP [meter/second]": as1(US2),
        "VSHIP [meter/second]": as1(VS2),
        "HEADING [degrees]": as1(HD2),
        "PITCH [degrees]": as1(PI2),
        "ROLL [degrees]": as1(RO2),
        # QC flags
        "TIME_SEADATANET_QC": (q_time if q_time is None else as1(ensure_2d_from_1d(q_time, (MAXT, MAXZ)))),
        "POSITION_SEADATANET_QC": (q_pos if q_pos is None else as1(ensure_2d_from_1d(q_pos, (MAXT, MAXZ)))),
        "PROFZ_SEADATANET_QC": as1(q_profz),
        "UCUR_SEADATANET_QC": as1(q_u),
        "VCUR_SEADATANET_QC": as1(q_v),
        "WCUR_SEADATANET_QC": as1(q_w),
        "USHIP_SEADATANET_QC": (q_ush if q_ush is None else as1(ensure_2d_from_1d(q_ush, (MAXT, MAXZ)))),
        "VSHIP_SEADATANET_QC": (q_vsh if q_vsh is None else as1(ensure_2d_from_1d(q_vsh, (MAXT, MAXZ)))),
        "HEADING_SEADATANET_QC": (q_head if q_head is None else as1(ensure_2d_from_1d(q_head, (MAXT, MAXZ)))),
        "PITCH_SEADATANET_QC": (q_pitch if q_pitch is None else as1(ensure_2d_from_1d(q_pitch, (MAXT, MAXZ)))),
        "ROLL_SEADATANET_QC": (q_roll if q_roll is None else as1(ensure_2d_from_1d(q_roll, (MAXT, MAXZ)))),
        "BOTTOM_DEPTH_SEADATANET_QC": (q_bot if q_bot is None else as1(ensure_2d_from_1d(q_bot, (MAXT, MAXZ)))),
        "BATHY_SEADATANET_QC": (q_bathy if q_bathy is None else as1(ensure_2d_from_1d(q_bathy, (MAXT, MAXZ)))),
        "Source_File": os.path.basename(fp),            # <- lightweight per-file source tag
        "LOCAL_CDI_ID": cdi
    })

    ds.close()
    return df

def main():
    nc_files = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.nc"), recursive=True))
    if not nc_files:
        raise SystemExit(f"No .nc files under: {INPUT_DIR}")

    rows = []
    for fp in nc_files:
        try:
            df = process_file(fp)
            rows.append(df)
            print(f"OK: {os.path.basename(fp)} -> {len(df):,} rows")
        except Exception as e:
            warnings.warn(f"Failed to parse {fp}: {e}")

    if not rows:
        raise SystemExit("No rows parsed; check INPUT_DIR and variable names.")

    # NOTE: avoid huge intermediate memory spikes; concat once
    big = pd.concat(rows, ignore_index=True)

    # Sort by Cruise, Station (time encoded), then depth
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = pd.to_datetime(big["yyyy-mm-ddThh:mm:ss.sss"], errors="coerce")
    big = big.assign(_t=ts).sort_values(["Cruise","Station","_t","PROFZ [meters]"]).drop(columns=["_t"])

    # Write ODV Generic Spreadsheet
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("// ODV Spreadsheet: SADCP trajectoryProfile merger (per-ensemble stations; full resolution)\n")
        f.write("// Note: ECUR is VELOCITY ERROR; SPEED = sqrt(UCUR^2 + VCUR^2)\n")
        f.write("\t".join(big.columns) + "\n")
        big.to_csv(f, sep="\t", index=False, header=False, float_format="%.6f", na_rep="")

    print(f"Saved: {OUTPUT_PATH}  ({len(big):,} rows)")

if __name__ == "__main__":
    main()
