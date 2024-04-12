if __name__ == "__main__":
    import json
    import os
    import sys

    root_path = "local"
    region = "ncedc"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    config = {
        "minlatitude": 32,
        "maxlatitude": 43,
        "minlongitude": -126.0,
        "maxlongitude": -114.0,
        "num_nodes": 1,
        "sampling_rate": 100,
        "degree2km": 111.1949,
        "channel": "HH*,BH*,EH*,HN*",
    }

    with open(f"{root_path}/{region}/config.json", "w") as fp:
        json.dump(config, fp, indent=2)