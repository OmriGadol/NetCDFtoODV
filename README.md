Copernicus CTD matcher
This repository contains a Python script that reads CTD ODV .text files and Copernicus model NetCDF outputs. It interpolates temperature and salinity at each CTD station's location, date, and depths, and writes a single combined ODV-compatible output file.
 
Repository Structure
•	copernicus_ctd_matcher5.py
Main script that:
1.	Parses an existing CTD ODV file
2.	Loads Copernicus NetCDF model data (temperature and salinity)
3.	Bilinearly interpolates model values at each CTD location & depth
4.	Optionally processes manual station definitions
5.	Writes a combined ODV-compatible text file containing model-only data, with Cruise and LOCAL_CDI_ID prefixed by Model_.
•	README.md
This file includes: overview, requirements, configuration, and usage instructions.
 
Requirements
•	Python 3.8 or higher
•	Packages:
o	numpy
o	pandas
o	xarray
 
Configuration
1.	User Options (at top of generate_model_only_fullODV.py):
o	use_ctd (bool): include stations from the CTD file
o	use_manual (bool): include stations defined manually
o	manual_stations (list of dicts): each dict with keys:
	Station, Cruise, LOCAL_CDI_ID
	datetime (string, e.g. 2013-10-23T07:00:00.000)
	Latitude [degrees_north], Longitude [degrees_east]
	PRESSURES (list of dbar levels)
2.	File Paths:
o	ctd_file: path to the input CTD ODV text file
o	temp_nc: path to Copernicus temperature NetCDF
o	sal_nc: path to Copernicus salinity NetCDF
o	output_file: desired path for the combined output
 
Usage
1.	Modify the USER OPTIONS section in generate_model_only_fullODV.py to suit your needs.
2.	Adjust the FILE PATHS to point at your CTD file and NetCDF files.
3.	Run the script:
4.	python generate_model_only_fullODV.py
5.	The script writes the combined ODV text file to output_file, ready to load into ODV.
 
Output
•	A single tab-delimited file (`all_model_only_fullODV.txt` by default) containing the original CTD column structure and header metadata, but with **PSAL** and **TEMP** replaced by interpolated model values at each station and depth.
**Cruise** and **LOCAL_CDI_ID** fields in the output are prefixed with `Model_` to clearly distinguish model-derived records.

 
![image](https://github.com/user-attachments/assets/3b4b0135-4705-40f8-982f-996690ecf37f)
