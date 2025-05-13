# Copernicus CTD Matcher

**Version:** 1.0.1  
**Date:** 2025-05-XX  
**Author:** OpenAI ChatGPT

---

## Overview

`Copernicus_ctd_matcher.py`  
Reads CTD casts (from an ODV-formatted file, CSV, or manual list), samples Copernicus model temperature (`thetao`) and salinity (`so`) at each station’s location and depth(s), and writes a single ODV-compatible output containing **only** the model-derived casts.  

Optional features (toggle via flags):  
- **Bathymetry sampling** from GeoTIFF  
- **Vertical interpolation** for fine depth grids  
- **Running-mean smoothing** of profiles  
- **Pycnocline detection** and **“sharpening”** around the strongest density gradient  

---

## Dependencies

- Python 3.8+  
- pandas  
- xarray  
- numpy  
- rasterio  
- gsw (TEOS‑10 routines)

Install with:

```bash
pip install pandas xarray numpy rasterio gsw
```

---

## Repository Structure

```
/ (project root)
├── Copernicus_ctd_matcher.py        # Main script
├── README.md                        # This file
├── requirements.txt                 # Dependency list
└── data/                            # Example inputs
```

---

## Usage

```bash
python Copernicus_ctd_matcher.py
```

---

## Configuration

All options live near the top of `Copernicus_ctd_matcher.py` under **USER OPTIONS**.

### File paths

```python
ctd_file        = '/path/to/original_ctd.txt'
temp_nc         = '/path/to/temperature_model.nc'
sal_nc          = '/path/to/salinity_model.nc'
output_file     = '/path/to/output_fullODV.txt'
bathy_tif       = '/path/to/bathymetry.tif'
manual_csv_path = '/path/to/points_list.csv'
```

### Station input sources

```python
use_ctd        = True    # read casts from CTD file
use_manual     = False   # append hard-coded `manual_stations`
use_manual_csv = False   # read station records from CSV
```

### Bathymetry

```python
use_bathy = True    # sample bottom depth from bathymetry GeoTIFF
```

### Vertical interpolation

```python
apply_vertical_interp = True   # upsample depths via linear interp
vertical_levels       = 50     # number of fine depth levels
```

### Smoothing

```python
apply_smoothing  = False  # apply running-mean filter
smoothing_window = 5      # window size
```

### Pycnocline detection & sharpening

```python
compute_pycnocline = False  # find the depth of maximum dρ/dz
sharpen_pyc        = False  # add extra levels around detected pycnocline
pyc_window         = 5.0    # ± meters around pycnocline for extra levels
pyc_npoints        = 7      # number of extra depth points in window
```

### Manual stations (if `use_manual=True`)

```python
manual_stations = [
    {
      'Station': 'MAN1',
      'Cruise': 'CustomCast1',
      'LOCAL_CDI_ID': 'Manual001',
      'datetime': '2013-10-23T07:00:00.000',
      'Latitude [degrees_north]': 32.85,
      'Longitude [degrees_east]': 35.02,
      'PRESSURES': [1, 10, 20, 50, 100]
    },
]
```

### CSV format (if `use_manual_csv=True`)

| Station | Cruise      | LOCAL_CDI_ID | datetime               | Latitude [°N] | Longitude [°E] | PRESSURES       |
|---------|-------------|--------------|------------------------|---------------|----------------|-----------------|
| MAN1    | CustomCast1 | Manual001    | 2013-10-23T07:00:00.000| 32.85         | 35.02          | 1;10;20;50;100  |

- `PRESSURES` may be comma- or semicolon-separated.

---

## Output

- Preserves header & column order from original CTD file  
- Prefixes `Cruise` & `LOCAL_CDI_ID` with `Model_`  
- Replaces PSAL & TEMP with model values  
- Overwrites `Bot. Depth [m]` if bathymetry sampling is enabled  
- Skips any station out of model bounds (reports at end)  

---

## License

This script is provided under the MIT License.  
