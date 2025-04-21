import xarray as xr
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# File paths (update these paths to actual file locations as needed)
ctd_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/haifa_uni_05.txt'  # Input CTD ODV file
model_sal_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/med-cmcc-sal-rean-m_1744205630998.nc'  # Copernicus salinity (so) NetCDF
model_temp_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/med-cmcc-tem-rean-m_1744206250724.nc'  # Copernicus temperature (thetao) NetCDF
output_file = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/copernicus_sampled_ctd.txt"  # Output path

# 1. Read and parse the CTD ODV file
with open(ctd_file, 'r') as f:
    lines = f.readlines()

# Separate header and data lines
header_lines = []
data_lines = []
in_header = True
for line in lines:
    if in_header:
        header_lines.append(line)
        if line.strip().startswith("Cruise"):
            pass
    else:
        data_lines.append(line)

# Extract the station data
header_end_idx = 0
for i, line in enumerate(lines):
    if line.startswith("//"):
        continue
    if not line.startswith("//") and line.strip() != "" and not line.strip().startswith("Cruise"):
        header_end_idx = i - 1
        break

header_lines = lines[:header_end_idx+1]
data_lines = lines[header_end_idx+1:]

# Write header lines directly to output
output_lines = header_lines.copy()

# Open the Copernicus model datasets
sal_ds = xr.open_dataset(model_sal_file)
temp_ds = xr.open_dataset(model_temp_file)
sal_var = 'so'       # Salinity variable name in dataset
temp_var = 'thetao'  # Temperature variable name in dataset

# Convert model time to datetime objects for easy comparison
model_times = sal_ds['time'].values
if np.issubdtype(model_times.dtype, np.datetime64):
    model_time_datetimes = model_times.astype('datetime64[ns]').astype('datetime64[s]').astype(object)
else:
    model_time_datetimes = sal_ds['time'].to_index().to_pydatetime()

# Process station data and sample Copernicus data
i = 0
station_count = 0
while i < len(data_lines):
    line = data_lines[i]
    if line.strip() == "":  # skip empty lines if any
        i += 1
        continue

    # Identify a station header line
    if not line[0].isspace():
        # Combine with next line if station metadata is split across lines
        station_line = line.strip("\n")
        next_line = data_lines[i+1] if i+1 < len(data_lines) else ""
        if next_line and not next_line[0].isspace():
            station_line += " " + next_line.strip("\n")
            i += 1
        tokens = station_line.split()
        if len(tokens) < 9:
            logger.warning("Station line parsing issue: expected at least 9 tokens, got %d tokens: %s", len(tokens), tokens)
            i += 1
            continue

        # Extract station metadata
        cruise = tokens[0]; station = tokens[1]; st_type = tokens[2]
        datetime_str = tokens[3]
        lon = float(tokens[4]); lat = float(tokens[5])
        local_id = tokens[6]; edmo_code = tokens[7]
        try:
            bot_depth = float(tokens[8])
        except ValueError:
            bot_depth = None

        # Convert datetime string to Python datetime
        try:
            ctd_time = datetime.fromisoformat(datetime_str)
        except Exception as e:
            try:
                ctd_time = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
            except:
                ctd_time = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")

        # Find nearest model time index to ctd_time
        ctd_time_ts = np.datetime64(ctd_time)
        time_deltas = [abs((t - ctd_time).total_seconds()) for t in model_time_datetimes]
        nearest_idx = int(np.argmin(time_deltas))
        nearest_model_time = model_time_datetimes[nearest_idx]
        if time_deltas[nearest_idx] > 3600*24*5:
            logger.warning(f"Station {station}: CTD time {ctd_time} is far from model time {nearest_model_time}")
        logger.info(f"Station {station}: CTD time {ctd_time}, matched model time {nearest_model_time}")

        # Select the model data at the nearest time
        sal_slice = sal_ds[sal_var].isel(time=nearest_idx)
        temp_slice = temp_ds[temp_var].isel(time=nearest_idx)

        # Prepare to collect output lines for this station
        station_output_lines = []
        first_line_meta = f"{cruise}\t{station}\t{st_type}\t{datetime_str}\t{lon:.6f}\t{lat:.6f}"

        # Now process the data records for this station
        data_tokens_list = []
        if len(tokens) > 9:
            first_data_tokens = tokens[9:]
            data_tokens_list.append(first_data_tokens)

        j = i + 1
        while j < len(data_lines) and data_lines[j][0].isspace():
            data_line = data_lines[j].strip()
            if data_line == "":
                j += 1
                continue
            data_tokens = data_line.split()
            data_tokens_list.append(data_tokens)
            j += 1

        i = j  # move index to next station

        logger.info(f"Station {station}: {len(data_tokens_list)} depth levels")

        for k, tokens_val in enumerate(data_tokens_list):
            try:
                depth_val = float(tokens_val[0])
            except ValueError:
                depth_val = np.nan
            if np.isnan(depth_val):
                model_sal_val = np.nan
                model_temp_val = np.nan
            else:
                model_point = {'latitude': lat, 'longitude': lon, 'depth': depth_val}
                try:
                    model_sal_val = float(sal_slice.interp(**model_point))
                    model_temp_val = float(temp_slice.interp(**model_point))
                except Exception as e:
                    model_sal_val = float(sal_slice.interp(latitude=lat, longitude=lon, depth=depth_val))
                    model_temp_val = float(temp_slice.interp(latitude=lat, longitude=lon, depth=depth_val))

            if np.isnan(model_sal_val) or np.isnan(model_temp_val):
                logger.warning(f" No model data for Station {station} at depth {depth_val}. Keeping original.")
                new_psal_str = tokens_val[2] if len(tokens_val) > 2 else ""
                new_temp_str = tokens_val[4] if len(tokens_val) > 4 else ""
            else:
                new_psal_str = f"{model_sal_val:.4f}"
                new_temp_str = f"{model_temp_val:.4f}"

            if len(tokens_val) > 2:
                tokens_val[2] = new_psal_str
            if len(tokens_val) > 4:
                tokens_val[4] = new_temp_str

            data_tokens_list[k] = tokens_val

        if data_tokens_list:
            first_data_tokens = data_tokens_list[0]
            meta2 = f"{local_id}\t{edmo_code}"
            meta3 = f"{bot_depth:.2f}" if bot_depth is not None else ""
            first_data_str = " ".join(first_data_tokens)
            first_line_full = f"{first_line_meta}\t{meta2}\t{meta3}\t{first_data_str}"
            station_output_lines.append(first_line_full)
            for tokens_val in data_tokens_list[1:]:
                data_str = " ".join(tokens_val)
                station_output_lines.append("\t" * 4 + data_str)

        output_lines.extend(line + "\n" for line in station_output_lines)
        station_count += 1
    else:
        i += 1

logger.info(f"Processed {station_count} stations. Writing output file.")
with open(output_file, 'w') as fout:
    for line in output_lines:
        fout.write(line)
