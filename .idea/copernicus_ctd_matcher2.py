import xarray as xr
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# File paths (update these paths to actual file locations as needed)
ctd_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/ODV/haifa_uni_05.txt'            # Input CTD ODV file
model_sal_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/med-cmcc-sal-rean-m_1744205630998.nc'  # Copernicus salinity (so) NetCDF
model_temp_file = '/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/med-cmcc-tem-rean-m_1744206250724.nc'  # Copernicus temperature (thetao) NetCDF
output_file = "/Users/ogado/Library/CloudStorage/OneDrive-UniversidadedeLisboa/GEM-SBP/Oceanogrphic_data/Copernicus/copernicus_sampled_ctd.txt"  # Use "." for current folder or specify full path like "outputs/"

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
        # The header section ends when we hit the column header line (starting with "Cruise")
        # We know the actual data starts right after the line that contains "QV:SEADATANET   IRRDUV" (last header line in this file).
        # We'll detect transition by finding the first line that looks like a station entry.
        if line.strip().startswith("Cruise"):
            # The next few lines are column names and units; find end of header when line doesn't contain QV or units and is not comment.
            # Alternatively, we continue to include header until we see the first actual data line (which starts with a cruise name).
            pass
    else:
        data_lines.append(line)
# Determine where header ends and data begins:
# Find index of first station line by skipping initial comments and header lines.
header_end_idx = 0
for i, line in enumerate(lines):
    # Comment lines start with "//"
    if line.startswith("//"):
        continue
    # The first non-comment line starting with a Cruise name (not "Cruise" header) is the first station.
    if not line.startswith("//") and line.strip() != "" and not line.strip().startswith("Cruise"):
        header_end_idx = i - 1  # last header line index
        break

header_lines = lines[:header_end_idx+1]
data_lines = lines[header_end_idx+1:]

# Write header lines directly to output (we will write to file at the end).
output_lines = header_lines.copy()

# Open the Copernicus model datasets
sal_ds = xr.open_dataset(model_sal_file)
temp_ds = xr.open_dataset(model_temp_file)
# Ensure we have the correct variable names
sal_var = 'so'       # salinity variable name in dataset
temp_var = 'thetao'  # temperature variable name in dataset

# Convert model time to datetime objects for easy comparison (if not already)
# We assume time coordinate is either datetime64 or numeric (in which case xarray will handle conversion if it has units).
model_times = sal_ds['time'].values
# If model_times are numpy datetime64, convert to Python datetime for easier subtraction
if np.issubdtype(model_times.dtype, np.datetime64):
    model_time_datetimes = model_times.astype('datetime64[ns]').astype('datetime64[s]').astype(object)
else:
    # If times are numeric (e.g., days since reference), let xarray convert via to_index()
    model_time_datetimes = sal_ds['time'].to_index().to_pydatetime()

# Loop through station data lines and process each station
i = 0
station_count = 0
while i < len(data_lines):
    line = data_lines[i]
    if line.strip() == "":  # skip empty lines if any
        i += 1
        continue
    # Identify a station header line (first line of a station block):
    # It should not start with whitespace (meaning metadata columns present).
    if not line[0].isspace():
        # Combine with next line if station metadata is split across lines
        station_line = line.strip("\n")
        next_line = data_lines[i+1] if i+1 < len(data_lines) else ""
        # Heuristic: if next line does NOT start with a lot of spaces (i.e., not a fully indented data line),
        # and does not start with a new cruise name, then it is likely the continuation of station header.
        if next_line and not next_line[0].isspace():
            station_line += " " + next_line.strip("\n")
            i += 1  # skip the next line as it was part of header
        # Parse station_line tokens
        tokens = station_line.split()
        if len(tokens) < 9:
            logger.warning("Station line parsing issue: expected at least 9 tokens, got %d tokens: %s", len(tokens), tokens)
            i += 1
            continue

        # Extract station metadata
        cruise = tokens[0]; station = tokens[1]; st_type = tokens[2]
        datetime_str = tokens[3]  # already in ISO format, e.g., "2013-10-23T06:41:00.00"
        lon = float(tokens[4]); lat = float(tokens[5])
        local_id = tokens[6]; edmo_code = tokens[7]
        try:
            bot_depth = float(tokens[8])
        except ValueError:
            bot_depth = None  # in case of missing bottom depth

        # Convert datetime string to Python datetime for model time matching
        try:
            ctd_time = datetime.fromisoformat(datetime_str)
        except Exception as e:
            # If fromisoformat fails (due to the format with milliseconds), use datetime.strptime
            try:
                ctd_time = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
            except:
                ctd_time = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")

        # Find nearest model time index to ctd_time
        ctd_time_ts = np.datetime64(ctd_time)
        # Compute time differences in absolute seconds
        time_deltas = [abs((t - ctd_time).total_seconds()) for t in model_time_datetimes]
        nearest_idx = int(np.argmin(time_deltas))
        nearest_model_time = model_time_datetimes[nearest_idx]
        if time_deltas[nearest_idx] > 3600*24*5:
            # If nearest model time is more than 5 days away, log a warning
            logger.warning(f"Station {station}: CTD time {ctd_time} is far from model time {nearest_model_time}")
        logger.info(f"Station {station}: CTD time {ctd_time}, matched model time {nearest_model_time}")

        # Select the model data at the nearest time (this gives a 3D field for that time)
        sal_slice = sal_ds[sal_var].isel(time=nearest_idx)
        temp_slice = temp_ds[temp_var].isel(time=nearest_idx)

        # Prepare to collect output lines for this station
        station_output_lines = []
        # Construct the first output line (station meta + possibly part of first data)
        # We'll fill this after processing the first data record.
        first_line_meta = f"{cruise}\t{station}\t{st_type}\t{datetime_str}\t{lon:.6f}\t{lat:.6f}"
        # (Using tab or single space as delimiter; here tabs for readability. We will adjust spacing later if needed.)

        # Now process the data records for this station
        data_tokens_list = []  # will hold token lists for each data record
        # The first data record is already in tokens[9:] of station_line (if present)
        if len(tokens) > 9:
            first_data_tokens = tokens[9:]
            data_tokens_list.append(first_data_tokens)
        # Process subsequent data lines (indented lines) until next station
        j = i + 1
        while j < len(data_lines) and data_lines[j][0].isspace():
            data_line = data_lines[j].strip()
            if data_line == "":
                j += 1
                continue
            data_tokens = data_line.split()
            data_tokens_list.append(data_tokens)
            j += 1

        # Now data_tokens_list contains all data records (each a list of strings) for this station
        i = j  # move index to start of next station block for the outer loop

        logger.info(f"Station {station}: {len(data_tokens_list)} depth levels")

        # Interpolate and replace values for each data record
        for k, tokens_val in enumerate(data_tokens_list):
            try:
                # Pressure/Depth value (assuming it's the first token in data line tokens)
                depth_val = float(tokens_val[0])
            except ValueError:
                # If not a valid float (e.g., missing), skip
                depth_val = np.nan
            # Get model values via interpolation (if depth is within model range)
            if np.isnan(depth_val):
                model_sal_val = np.nan
                model_temp_val = np.nan
            else:
                model_point = {'latitude': lat, 'longitude': lon, 'depth': depth_val}
                # Perform linear interpolation in space for salinity & temp
                try:
                    model_sal_val = float(sal_slice.interp(**model_point))
                    model_temp_val = float(temp_slice.interp(**model_point))
                except Exception as e:
                    model_sal_val = float(sal_slice.interp(latitude=lat, longitude=lon, depth=depth_val))
                    model_temp_val = float(temp_slice.interp(latitude=lat, longitude=lon, depth=depth_val))
            # If out-of-range (xarray returns nan)
            if np.isnan(model_sal_val) or np.isnan(model_temp_val):
                logger.warning(f" No model data for Station {station} at depth {depth_val}. Keeping original.")
                # We keep original values (which are tokens_val[2] for PSAL, [4] for TEMP)
                new_psal_str = tokens_val[2] if len(tokens_val) > 2 else ""
                new_temp_str = tokens_val[4] if len(tokens_val) > 4 else ""
            else:
                # Format new values to match original precision (assuming 4 decimal places for both PSAL and TEMP)
                new_psal_str = f"{model_sal_val:.4f}"
                new_temp_str = f"{model_temp_val:.4f}"
            # Replace PSAL and TEMP tokens in the token list
            if len(tokens_val) > 2:
                tokens_val[2] = new_psal_str
            if len(tokens_val) > 4:
                tokens_val[4] = new_temp_str
            # (We leave QV tokens as-is. Optionally, set tokens_val[3] and [5] to '0' or another flag to indicate substitution.)
            data_tokens_list[k] = tokens_val

        # Reconstruct the station's output lines with proper spacing.
        # We will maintain the same number of tokens per line as original, inserting spaces as needed.
        # First, split the first data record from metadata for output:
        if data_tokens_list:
            # Build second part of first line: Local_ID, EDMO, Bottom depth, then first data tokens
            first_data_tokens = data_tokens_list[0]
            # Compose the remaining of first line
            meta2 = f"{local_id}\t{edmo_code}"
            meta3 = f"{bot_depth:.2f}" if bot_depth is not None else ""
            first_data_str = " ".join(first_data_tokens)
            first_line_full = f"{first_line_meta}\t{meta2}\t{meta3}\t{first_data_str}"
            station_output_lines.append(first_line_full)
            # Now subsequent lines (2nd data record onwards) - these start with no metadata, so just indent or blank tab for alignment
            for tokens_val in data_tokens_list[1:]:
                data_str = " ".join(tokens_val)
                # We can prepend a tab or spaces to align under PRES column. For simplicity, use a tab here.
                station_output_lines.append("\t" * 4 + data_str)  # using 4 tabs as an approximation of indent
        else:
            # No data tokens (should not happen for CTD), but handle gracefully
            station_output_lines.append(first_line_meta + "\t" + local_id + "\t" + str(edmo_code) + "\t" + (str(bot_depth) if bot_depth else ""))

        # Add this station's lines to output_lines
        output_lines.extend(line + "\n" for line in station_output_lines)
        station_count += 1
    else:
        # If line starts with whitespace but we haven't identified a station (shouldn't happen here), skip.
        i += 1

logger.info(f"Processed {station_count} stations. Writing output file.")
# Write output lines to file
with open(output_file, 'w') as fout:
    for line in output_lines:
        fout.write(line)
