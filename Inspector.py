#!/usr/bin/env python3
"""
Inspect a NetCDF ADCP file and print its structure, variable names,
dimensions, units, standard_name and long_name.

Requires: xarray, netCDF4
"""

import xarray as xr

# ------------------ HARD-CODED FILE PATH ------------------
FILE_PATH = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/Cadiz/CurrentMeter/order_74890_unrestricted/nc/000486_CFPOINT_SADCP_10030190_203625-1_V0.nc"
# ----------------------------------------------------------

print(f"\n🔍 Inspecting NetCDF file:\n{FILE_PATH}\n")

try:
    ds = xr.open_dataset(FILE_PATH, decode_times=False)
except Exception as e:
    print(f"❌ Failed to open file: {e}")
    raise SystemExit(1)

# ---- Global attributes ----
print("=== 🌍 Global attributes ===")
for k, v in ds.attrs.items():
    print(f"{k}: {v}")

# ---- Dimensions ----
print("\n=== 📏 Dimensions ===")
for dim, size in ds.dims.items():
    print(f"{dim}: {size}")

# ---- Variables ----
print("\n=== 📊 Variables ===")
for var_name, var in ds.variables.items():
    dims = ", ".join(var.dims)
    units = var.attrs.get("units", "")
    std = var.attrs.get("standard_name", "")
    long = var.attrs.get("long_name", "")
    print(f"{var_name:>30} | dims=({dims:20}) | units={units:20} | standard_name={std:30} | long_name={long}")

# ---- Coordinates ----
print("\n=== 📌 Coordinates ===")
for coord in ds.coords:
    print(coord)

ds.close()
print("\n✅ Inspection complete.\n")
