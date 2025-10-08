import subprocess
import os

def main():
    # — USER CREDENTIALS —
    USER = os.environ.get("CMEMS_USER", "ogadol")
    PWD = os.environ.get("CMEMS_PWD", "Elmida2503")

    # Subset parameters
    motu = "https://nrt.cmems-du.eu/motu-web/Motu"
    service_id = "MEDSEA_DAILY_PHY_006_004-TDS"
    product_id = "dataset-du-medsea-daily"
    lon_min, lon_max = 34.5, 35.5
    lat_min, lat_max = 32.5, 33.5
    depth_min, depth_max = 0.5, 200.0
    start_date = "2013-10-23 00:00:00"
    end_date = "2013-10-24 00:00:00"
    variables = ["thetao", "so"]
    out_name = "medsea_daily_subset_20131023.nc"
    out_dir = "."

    # Path to the motuclient executable
    motuclient_path = os.path.join(
        "/Users", "ogado", "PycharmProjects", "NetCDf", ".venv", "bin", "motuclient"
    )

    # Build the CLI command
    command = [
        motuclient_path,
        "--motu", motu,
        "--service-id", service_id,
        "--product-id", product_id,
        "--longitude-min", str(lon_min),
        "--longitude-max", str(lon_max),
        "--latitude-min", str(lat_min),
        "--latitude-max", str(lat_max),
        "--depth-min", str(depth_min),
        "--depth-max", str(depth_max),
        "--date-min", start_date,
        "--date-max", end_date,
        "--variable", *variables,
        "--out-dir", out_dir,
        "--out-name", out_name,
        "--user", USER,
        "--pwd", PWD
    ]

    # Execute the command
    try:
        print("⏳ Requesting subset...")
        subprocess.run(command, check=True)
        print("✅ Download complete!")
        print(f"Your subset is stored in `{os.path.join(out_dir, out_name)}`.")
    except subprocess.CalledProcessError as e:
        print("❌ An error occurred while downloading the data.")
        print(e)

if __name__ == "__main__":
    # Set environment variables explicitly
    os.environ['CMEMS_USER'] = 'ogadol'
    os.environ['CMEMS_PWD'] = 'Elmida2503'
    main()