import os
import fsspec

# Configuration for NC -> NCEDC
configs = [
    {
        "input_bucket": "quakeflow_catalog",
        "input_folder": "NC/FDSNstationXML",
        "output_bucket": "quakeflow_dataset",
        "output_folder": "NCEDC/FDSNstationXML",
    },
    {
        "input_bucket": "quakeflow_catalog",
        "input_folder": "SC/FDSNstationXML",
        "output_bucket": "quakeflow_dataset",
        "output_folder": "SCEDC/FDSNstationXML",
    },
]

protocol = "gs"
token = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"

fs = fsspec.filesystem(protocol, token=token)

for config in configs:
    input_bucket = config["input_bucket"]
    input_folder = config["input_folder"]
    output_bucket = config["output_bucket"]
    output_folder = config["output_folder"]

    print(f"\nProcessing {input_bucket}/{input_folder} -> {output_bucket}/{output_folder}")

    # List all input files
    input_path = f"{input_bucket}/{input_folder}"
    input_files = fs.glob(f"{input_path}/**/*.xml")
    print(f"Found {len(input_files)} input files")

    # List all output files
    output_path = f"{output_bucket}/{output_folder}"
    try:
        output_files = set(fs.glob(f"{output_path}/**/*.xml"))
    except:
        output_files = set()
    print(f"Found {len(output_files)} existing output files")

    # Copy missing files
    copied = 0
    skipped = 0
    renamed = 0
    for input_file in input_files:
        # Get relative path from input folder
        rel_path = input_file.replace(f"{input_bucket}/{input_folder}/", "")

        # Rename if not following network.station.xml convention (e.g., network_station.xml)
        filename = os.path.basename(rel_path)
        dirname = os.path.dirname(rel_path)
        if "_" in filename and "." not in filename.replace(".xml", ""):
            # Convert network_station.xml to network.station.xml
            new_filename = filename.replace("_", ".", 1)
            rel_path = os.path.join(dirname, new_filename)
            renamed += 1

        output_file = f"{output_bucket}/{output_folder}/{rel_path}"

        if output_file in output_files:
            skipped += 1
            continue

        # Copy file
        print(f"Copying {os.path.basename(input_file)} -> {rel_path}")
        fs.copy(input_file, output_file)
        copied += 1

    print(f"Copied {copied} files ({renamed} renamed), skipped {skipped} existing files")
