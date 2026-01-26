# %%
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
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
if __name__ == "__main__":
    copy_xml_to_gcs(overwrite=True, max_workers=16)

# %%
