# Copernicus_ctd_matcher

**Version:** 1.0.0  
**Date:** 2025-04-19  
**Author:** OpenAI ChatGPT

---

## Overview

`Copernicus_ctd_matcher.py` reads a CTD ODV-format text file and Copernicus model NetCDF outputs (temperature & salinity), interpolates the model at each CTD station’s location and depth(s), and writes a combined ODV-compatible output file containing **only** the model-derived casts.

Key capabilities:
- Process all casts from a CTD file or a user-defined list of manual stations  
- Skip any stations outside the model’s spatial/time bounds (non-fatal, summary printed)  
- Prefix `Cruise` & `LOCAL_CDI_ID` fields with `Model_` for clarity  
- **Optional vertical interpolation**: sample at a dense depth grid via linear interpolation  
- **Optional smoothing**: apply an N-point running mean to reduce step artifacts  

---

## Dependencies

- Python 3.8+  
- pandas  
- xarray  
- numpy  

Install via:

```bash
pip install pandas xarray numpy
```

---

## Repository Structure

```
/ (project root)
├── Copernicus_ctd_matcher.py        # Main script
├── README.md                        # This file
├── requirements.txt                 # Dependency list
└── data/                            # Example CTD & NetCDF inputs
```

---

## Usage

```bash
python Copernicus_ctd_matcher.py
```

By default, the script reads the CTD file and writes an output file as specified by `output_file` in the configuration.

---

## Configuration

All user-configurable options live near the top of `Copernicus_ctd_matcher.py` under **USER OPTIONS**.

### File Paths

```python
ctd_file    = '/path/to/original_ctd.txt'
temp_nc     = '/path/to/temperature_model.nc'
sal_nc      = '/path/to/salinity_model.nc'
output_file = '/path/to/all_model_only_fullODV.txt'
```

### Flags

```python
# Station sources
use_ctd               = True     # Read casts from CTD file
use_manual            = False    # Append manual_stations list


 ## Configuration

 All user-configurable options live near the top of `Copernicus_ctd_matcher.py` under **USER OPTIONS**.

### CSV Points Input

If you want to pull your station list from a simple CSV file instead of (or in addition to) the CTD file or hard-coded list, use these flags:

```python
use_ctd        = False        # skip CTD file entirely
use_manual     = False        # skip the built-in manual_stations list
use_manual_csv = True         # read only from CSV
manual_csv_path = 'path/to/your_points.csv'
```

Your CSV must have these columns:

| Station | Cruise      | LOCAL_CDI_ID | datetime               | Latitude [degrees_north] | Longitude [degrees_east] | PRESSURES       |
|---------|-------------|--------------|------------------------|--------------------------|--------------------------|-----------------|
| MAN1    | CustomCast1 | Manual001    | 2013-10-23T07:00:00.000| 32.85                    | 35.02                    | 1;10;20;50;100  |

– where PRESSURES is a semicolon-separated list of dbar levels.




# Vertical interpolation
apply_vertical_interp = True     # Upsample depths via linear interpolation
vertical_levels       = 50       # Number of fine-depth levels

# Smoothing
apply_smoothing       = False    # Apply running-mean filter
smoothing_window      = 5        # Window size for running mean
```

### Manual Stations

Define a list of dictionaries if `use_manual=True`:

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

---

## Output

- **Header & columns**: preserved from the original CTD file (including metadata lines)  
- **Cruise** and **LOCAL_CDI_ID**: prefixed with `Model_`  
- **PSAL** and **TEMP**: replaced by model-derived values  
- **Skipped stations**: any cast outside model bounds is omitted, with its station ID & reason printed at the end.

Default output filename: `all_model_only_fullODV.txt`.

---

## License

This script is provided under the MIT License.  
Feel free to modify and redistribute.
