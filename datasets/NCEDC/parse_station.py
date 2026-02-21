# %%
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from pathlib import Path

import fsspec
import obspy
import pandas as pd
from tqdm import tqdm

# %%
input_base_url = "https://ncedc.org/ftp/pub/doc"
input_folder = input_base_url

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"
output_folder = "NCEDC/FDSNstationXML"  # network/network.station.xml


# %%
def copy_single_file(xml_file, overwrite=False):
    """Copy a single XML file to GCS."""
    input_fs = fsspec.filesystem("https")
    output_fs = fsspec.filesystem(output_protocol, token=output_token)

    filename = Path(xml_file).name
    # Extract network and station from filename (e.g., BK.BRK.xml -> network=BK, station=BRK)
    parts = filename.replace(".xml", "").split(".")
    network, station = parts[0], parts[1]
    output_path = f"{output_bucket}/{output_folder}/{network}/{network}.{station}.xml"

    if not overwrite and output_fs.exists(output_path):
        return None

    with input_fs.open(xml_file, "rb") as src:
        output_fs.makedirs(f"{output_bucket}/{output_folder}/{network}", exist_ok=True)
        with output_fs.open(output_path, "wb") as dst:
            dst.write(src.read())

    return f"{network}.{station}.xml"


def copy_xml_to_gcs(overwrite=False, max_workers=16):
    """Copy FDSN station XML files from NCEDC FTP to Google Cloud Storage."""
    input_fs = fsspec.filesystem("https")

    # List all network folders (*.info/)
    network_folders = input_fs.ls(input_folder)
    network_folders = [f["name"].rstrip("/") for f in network_folders if f["name"].rstrip("/").endswith(".info")]

    # Collect all files to copy
    all_tasks = []
    for network_folder in tqdm(network_folders, desc="Scanning networks"):
        network = Path(network_folder).name.replace(".info", "")

        fdsn_folder = f"{network_folder}/{network}.FDSN.xml"
        if not input_fs.exists(fdsn_folder):
            continue

        xml_files = input_fs.ls(fdsn_folder)
        xml_files = [f["name"] for f in xml_files if f["name"].endswith(".xml")]
        all_tasks.extend(xml_files)

    print(f"Found {len(all_tasks)} XML files to process")

    # Copy files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(copy_single_file, xml_file, overwrite): xml_file
            for xml_file in all_tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying files"):
            future.result()


# %%
def parse_inventory_csv(inventory):
    """Parse an obspy Inventory into a DataFrame of channel-level station info."""
    channel_list = []
    for network in inventory:
        for station in network:
            for channel in station:
                sensitivity = None
                if channel.response is not None and channel.response.instrument_sensitivity is not None:
                    sensitivity = channel.response.instrument_sensitivity.value

                channel_list.append(
                    {
                        "network": network.code,
                        "station": station.code,
                        "location": channel.location_code,
                        "instrument": channel.code[:-1],
                        "component": channel.code[-1],
                        "channel": channel.code,
                        "longitude": channel.longitude,
                        "latitude": channel.latitude,
                        "elevation_m": channel.elevation,
                        "local_depth_m": channel.depth,
                        "depth_km": round(-channel.elevation / 1000, 4),
                        "begin_time": (
                            channel.start_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                            if channel.start_date is not None
                            else None
                        ),
                        "end_time": (
                            channel.end_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                            if channel.end_date is not None
                            else None
                        ),
                        "azimuth": channel.azimuth,
                        "dip": channel.dip,
                        "sensitivity": sensitivity,
                        "site": station.site.name if station.site is not None else "",
                        "sensor": channel.sensor.description if channel.sensor is not None else "",
                    }
                )

    return pd.DataFrame(channel_list)


def read_single_xml(xml_path):
    """Read a single XML file and return an obspy Inventory."""
    fs = fsspec.filesystem(output_protocol, token=output_token)
    try:
        with fs.open(xml_path, "rb") as f:
            return obspy.read_inventory(f)
    except Exception as e:
        print(f"Failed to read {xml_path}: {e}")
        return None


def parse_xml_to_csv(output_csv=None, max_workers=16):
    """Parse all station XML files from GCS into a single CSV file.

    Args:
        output_csv: Output CSV path on GCS. Defaults to {output_bucket}/NCEDC/stations.csv.
        max_workers: Number of parallel workers for reading XML files.
    """
    if output_csv is None:
        output_csv = f"{output_bucket}/NCEDC/stations.csv"

    fs = fsspec.filesystem(output_protocol, token=output_token)

    # List all XML files under the station XML folder
    xml_files = fs.glob(f"{output_bucket}/{output_folder}/*/*.xml")
    print(f"Found {len(xml_files)} XML files")

    # Read XML files in parallel
    inv = obspy.Inventory()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_single_xml, xml_file): xml_file for xml_file in xml_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading XML"):
            result = future.result()
            if result is not None:
                inv += result

    # Convert to CSV
    stations = parse_inventory_csv(inv)
    print(f"Parsed {len(stations)} channels from {len(xml_files)} XML files")

    with fs.open(output_csv, "w") as f:
        stations.to_csv(f, index=False)
    print(f"Saved to {output_csv}")

    return stations


# %%
if __name__ == "__main__":
    # copy_xml_to_gcs(overwrite=True, max_workers=16)
    parse_xml_to_csv(max_workers=16)

# %%
