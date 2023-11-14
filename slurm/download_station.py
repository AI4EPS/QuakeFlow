# %%
from typing import Dict

from kfp import dsl


@dsl.component(base_image="zhuwq0/quakeflow:latest")
def download_station(
    root_path: str, region: str, config: Dict, protocol: str = "file", bucket: str = "", token: Dict = None
):
    import json
    import os
    import time
    from collections import defaultdict
    from datetime import timezone

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import fsspec
    import matplotlib
    import matplotlib.pyplot as plt
    import obspy
    import obspy.clients.fdsn
    import pandas as pd
    import pyproj

    matplotlib.use("Agg")

    # %%
    fs = fsspec.filesystem(protocol, token=token)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    data_dir = f"{region}/obspy"
    if not os.path.exists(f"{root_path}/{data_dir}"):
        os.makedirs(f"{root_path}/{data_dir}")

    print(json.dumps(config, indent=4))

    proj = pyproj.Proj(
        f"+proj=sterea +lon_0={(config['minlongitude'] + config['maxlongitude'])/2} +lat_0={(config['minlatitude'] + config['maxlatitude'])/2} +units=km"
    )

    # %%
    if ("provider" in config) and (config["provider"] is not None):
        print("Downloading station response...")
        inventory = obspy.Inventory()
        for provider in config["provider"]:
            if os.path.exists(f"{root_path}/{data_dir}/inventory_{provider.lower()}.xml"):
                print(f"Loading existing {root_path}/{data_dir}/inventory_{provider.lower()}.xml")
                inventory += obspy.read_inventory(f"{root_path}/{data_dir}/inventory_{provider.lower()}.xml")
                continue
            client = obspy.clients.fdsn.Client(provider, timeout=1200)
            max_retry = 10
            retry = 0
            stations = None
            while retry < max_retry:
                try:
                    stations = client.get_stations(
                        network=config["network"] if "network" in config else None,
                        station=config["station"] if "station" in config else None,
                        starttime=config["starttime"],
                        endtime=config["endtime"],
                        minlatitude=config["minlatitude"],
                        maxlatitude=config["maxlatitude"],
                        minlongitude=config["minlongitude"],
                        maxlongitude=config["maxlongitude"],
                        channel=config["channel"] if "channel" in config else None,
                        level=config["level"] if "level" in config else "response",
                    )
                    break
                except Exception as e:
                    message = "The current client does not have a station service."
                    if str(e)[: len(message)] == message:
                        print(f"{provider} failed: {e}")
                        break
                    print(e)
                    retry += 1
                    time.sleep(10)
                    continue
            if retry == max_retry:
                print(f"Failed to download {provider} after {max_retry} retries.")
                continue

            if stations is not None:
                print(
                    f"Dowloaded {len([chn for net in stations for sta in net for chn in sta])} stations from {provider.lower()}"
                )
                stations.write(f"{root_path}/{data_dir}/inventory_{provider.lower()}.xml", format="STATIONXML")
                if protocol != "file":
                    fs.put(
                        f"{root_path}/{data_dir}/inventory_{provider.lower()}.xml",
                        f"{bucket}/{data_dir}/inventory_{provider.lower()}.xml",
                    )
                inventory += stations

        inventory.write(f"{root_path}/{data_dir}/inventory.xml", format="STATIONXML")
        if protocol != "file":
            fs.put(f"{root_path}/{data_dir}/inventory.xml", f"{bucket}/{data_dir}/inventory.xml")

    # %%

    def parse_inventory_json(inventory, mseed_ids=[]):
        comp = ["3", "2", "1", "U", "V", "E", "N", "Z"]
        order = {key: i for i, key in enumerate(comp)}
        stations = {}
        num = 0
        for net in inventory:
            for sta in net:
                sta_3c = {}
                components = defaultdict(list)

                for chn in sta:
                    key = f"{chn.location_code}.{chn.code[:-1]}"
                    components[key].append([chn.code[-1], chn.response.instrument_sensitivity.value])

                    if key not in sta_3c:
                        sta_3c[key] = {
                            "latitude": chn.latitude,
                            "longitude": chn.longitude,
                            "elevation_m": chn.elevation,
                            "location": chn.location_code,
                            "instrument": chn.code[:-1],
                        }

                for key in list(sta_3c.keys()):
                    station_id = f"{net.code}.{sta.code}.{sta_3c[key]['location']}.{sta_3c[key]['instrument']}"
                    if (len(mseed_ids) > 0) and (station_id not in mseed_ids):
                        continue
                    num += 1
                    x_km, y_km = proj(sta_3c[key]["longitude"], sta_3c[key]["latitude"])
                    z_km = -sta_3c[key]["elevation_m"] / 1e3
                    stations[station_id] = {
                        "network": net.code,
                        "station": sta.code,
                        "location": sta_3c[key]["location"],
                        "instrument": sta_3c[key]["instrument"],
                        "component": "".join([x[0] for x in sorted(components[key], key=lambda x: order[x[0]])]),
                        "sensitivity": [x[1] for x in sorted(components[key], key=lambda x: order[x[0]])],
                        "latitude": sta_3c[key]["latitude"],
                        "longitude": sta_3c[key]["longitude"],
                        "elevation_m": sta_3c[key]["elevation_m"],
                        "depth_km": -sta_3c[key]["elevation_m"] / 1e3,
                        "x_km": round(x_km, 3),
                        "y_km": round(y_km, 3),
                        "z_km": round(z_km, 3),
                    }

        print(f"Parse {num} stations of {provider} into json")

        return stations

    def parse_inventory_csv(inventory, mseed_ids=[]):
        channel_list = []
        for network in inventory:
            for station in network:
                for channel in station:
                    if channel.sensor is None:
                        sensor_description = ""
                    else:
                        sensor_description = channel.sensor.description
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
                            "depth_km": -channel.elevation / 1e3,
                            # "depth_km": channel.depth,
                            "begin_time": channel.start_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                            if channel.start_date is not None
                            else None,
                            "end_time": channel.end_date.datetime.replace(tzinfo=timezone.utc).isoformat()
                            if channel.end_date is not None
                            else None,
                            "azimuth": channel.azimuth,
                            "dip": channel.dip,
                            "sensitivity": channel.response.instrument_sensitivity.value,
                            "site": station.site.name,
                            "sensor": sensor_description,
                        }
                    )
        channel_list = pd.DataFrame(channel_list)

        print(f"Parse {len(channel_list)} channels of {provider} into csv")

        return channel_list

    for provider in config["provider"]:
        if os.path.exists(f"{root_path}/{data_dir}/inventory_{provider.lower()}.xml"):
            inventory = obspy.read_inventory(f"{root_path}/{data_dir}/inventory_{provider.lower()}.xml")
        stations = parse_inventory_csv(inventory)
        stations[["x_km", "y_km"]] = stations.apply(
            lambda row: pd.Series(proj(row["longitude"], row["latitude"])), axis=1
        )
        stations["z_km"] = stations["depth_km"]
        stations[["latitude", "longitude"]] = stations[["latitude", "longitude"]].round(4)
        stations["depth_km"] = stations["depth_km"].round(2)
        stations[["x_km", "y_km", "z_km"]] = stations[["x_km", "y_km", "z_km"]].round(2)
        stations = stations.sort_values(by=["network", "station", "location", "channel"])
        stations = stations.groupby(["network", "station", "location", "channel"]).first().reset_index()
        stations.to_csv(f"{root_path}/{data_dir}/stations_{provider.lower()}.csv", index=False)
        if protocol != "file":
            fs.put(
                f"{root_path}/{data_dir}/stations_{provider.lower()}.csv",
                f"{bucket}/{data_dir}/stations_{provider.lower()}.csv",
            )

        stations = parse_inventory_json(inventory)
        if len(stations) > 0:
            with open(f"{root_path}/{data_dir}/stations_{provider.lower()}.json", "w") as f:
                json.dump(stations, f, indent=4)
            if protocol != "file":
                fs.put(
                    f"{root_path}/{data_dir}/stations_{provider.lower()}.json",
                    f"{bucket}/{data_dir}/stations_{provider.lower()}.json",
                )

    # %% merge stations
    stations = []
    for provider in config["provider"]:
        tmp = pd.read_csv(f"{root_path}/{data_dir}/stations_{provider.lower()}.csv")
        tmp["provider"] = provider
        stations.append(tmp)
    stations = pd.concat(stations)
    stations = stations.groupby(["network", "station", "location", "channel"], dropna=False).first().reset_index()
    print(f"Merged {len(stations)} channels")
    stations.to_csv(f"{root_path}/{data_dir}/stations.csv", index=False)
    if protocol != "file":
        fs.put(f"{root_path}/{data_dir}/stations.csv", f"{bucket}/{data_dir}/stations.csv")

    stations = {}
    for provider in config["provider"]:
        with open(f"{root_path}/{data_dir}/stations_{provider.lower()}.json") as f:
            tmp = json.load(f)
        for key, value in tmp.items():
            if key not in stations:
                stations[key] = value
                stations[key]["provider"] = provider
    if len(stations) > 0:
        print(f"Merged {len(stations)} stations")
        with open(f"{root_path}/{data_dir}/stations.json", "w") as f:
            json.dump(stations, f, indent=4)
        if protocol != "file":
            fs.put(f"{root_path}/{data_dir}/stations.json", f"{bucket}/{data_dir}/stations.json")

    # %%
    def visulization(config, events=None, stations=None, fig_name="catalog.png"):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([config["minlongitude"], config["maxlongitude"], config["minlatitude"], config["maxlatitude"]])
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color="gray", alpha=0.5, linestyle="--"
        )
        if events is not None:
            if len(events) > 0:
                ax.scatter(
                    events.longitude,
                    events.latitude,
                    transform=ccrs.PlateCarree(),
                    s=1e4 / len(events),
                    c="r",
                    marker=".",
                    label="events",
                )
        if stations is not None:
            ax.scatter(
                stations.longitude,
                stations.latitude,
                transform=ccrs.PlateCarree(),
                s=100,
                c="b",
                marker="^",
                label="stations",
            )
        ax.legend(scatterpoints=1, markerscale=0.5, fontsize=10)
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")

    stations = pd.DataFrame.from_dict(stations, orient="index")
    events = None
    if os.path.exists(f"{root_path}/{data_dir}/events.csv"):
        events = pd.read_csv(f"{root_path}/{data_dir}/events.csv")
    visulization(config, events, stations, f"{root_path}/{data_dir}/stations.png")
    if protocol != "file":
        fs.put(f"{root_path}/{data_dir}/stations.png", f"{bucket}/{data_dir}/stations.png")


if __name__ == "__main__":
    import json
    import sys

    root_path = "local"
    region = "demo"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    download_station.python_func(
        root_path=root_path, region=region, config=config, protocol="file", bucket="", token=None
    )

    # # %%
    # import os

    # from kfp import compiler
    # from kfp.client import Client

    # bucket = "quakeflow_share"
    # protocol = "gs"
    # token = None
    # token_file = "/Users/weiqiang/.config/gcloud/application_default_credentials.json"
    # if os.path.exists(token_file):
    #     with open(token_file, "r") as fp:
    #         token = json.load(fp)

    # @dsl.pipeline
    # def test_download_station():
    #     download_station(
    #         root_path=root_path, region=region, config=config, protocol=protocol, bucket=bucket, token=token
    #     )

    # client = Client("3a1395ae1e4ad10-dot-us-west1.pipelines.googleusercontent.com")
    # run = client.create_run_from_pipeline_func(
    #     test_download_station,
    #     arguments={},
    # )
