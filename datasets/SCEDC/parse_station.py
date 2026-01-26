# %%
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
from tqdm import tqdm

# %%
input_protocol = "s3"
input_bucket = "scedc-pds"
input_folder = "FDSNstationXML"
input_fs = fsspec.filesystem(input_protocol, anon=True)

output_protocol = "gs"
output_token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
output_bucket = "quakeflow_dataset"
output_folder = "SCEDC/FDSNstationXML"
output_fs = fsspec.filesystem(output_protocol, token=output_token)


# %%
def copy_single_file(xml_file, overwrite=False):
    """Copy a single XML file to GCS."""
    input_fs = fsspec.filesystem(input_protocol, anon=True)
    output_fs = fsspec.filesystem(output_protocol, token=output_token)

    filename = Path(xml_file).name
    # Extract network and station from filename
    # Handle both formats: BC_AGSX.xml (underscore) and PB.B079.xml (dot)
    name = filename.replace(".xml", "")
    if "_" in name:
        network, station = name.split("_", 1)
    else:
        parts = name.split(".")
        network, station = parts[0], parts[1]
    output_path = f"{output_bucket}/{output_folder}/{network}/{network}.{station}.xml"

    if not overwrite and output_fs.exists(output_path):
        return None

    with input_fs.open(xml_file, "rb") as src:
        output_fs.makedirs(f"{output_bucket}/{output_folder}/{network}", exist_ok=True)
        with output_fs.open(output_path, "wb") as dst:
            dst.write(src.read())

    return f"{network}.{station}.xml"


def copy_xml_to_gcs(max_workers=16):
    """Copy FDSN station XML files from SCEDC S3 to Google Cloud Storage."""
    input_fs = fsspec.filesystem(input_protocol, anon=True)

    # List all folders (CI, unauthoritative-XML, etc.)
    folders = input_fs.ls(f"{input_bucket}/{input_folder}")
    # Separate unauthoritative and authoritative folders
    unauth_folders = [f for f in folders if "unauthoritative" in f]
    auth_folders = [f for f in folders if "unauthoritative" not in f]

    # Process unauthoritative files first, then authoritative files overwrite them
    for folder_group, group_name in [(unauth_folders, "unauthoritative"), (auth_folders, "authoritative")]:
        all_tasks = []
        for folder in folder_group:
            xml_files = input_fs.glob(f"{folder}/*.xml")
            all_tasks.extend(xml_files)

        if not all_tasks:
            continue

        print(f"Processing {len(all_tasks)} {group_name} XML files")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(copy_single_file, xml_file, overwrite=True): xml_file
                for xml_file in all_tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Copying {group_name}"):
                future.result()


# %%
if __name__ == "__main__":
    copy_xml_to_gcs(max_workers=16)

# %%
