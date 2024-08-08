# %%
from typing import Dict, List

from kfp import dsl


@dsl.component(base_image="zhuwq0/phasenet-api:latest")
def run_phasenet(
    root_path: str,
    region: str,
    config: Dict,
    rank: int = 0,
    model_path: str = "../PhaseNet/",
    mseed_list: List = None,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
) -> str:
    # %%
    import os
    from collections import defaultdict
    from glob import glob

    import fsspec

    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)

    # %%
    result_path = f"{region}/phasenet"
    if not os.path.exists(f"{root_path}/{result_path}"):
        os.makedirs(f"{root_path}/{result_path}")

    # %%
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        if protocol != "file":
            fs.get(f"{bucket}/{waveform_dir}/", f"{root_path}/{waveform_dir}/", recursive=True)

    # %%
    def mapping_mseed_path(
        data_path="",
        network="??",
        year="????",
        jday="???",
        month="??",
        day="??",
        hour="??",
        station="*",
        location="*",
        channel="*",
        datacenter="quakeflow",
        group_channel=False,
        protocol="file",
        anon=False,
        token=None,
    ):
        data_path = data_path.rstrip("/")
        if datacenter.lower() == "quakeflow":
            mseed_path = f"{data_path}/{year}-{jday}/{hour}/{network}.{station}.{location}.{channel}.mseed"
        elif datacenter.lower() == "ncedc":
            location = location.replace("-", "_")
            # mseed_path = f"s3://ncedc-pds/continuous_waveforms/{network}/{year}/{year}.{jday}/{station}.{network}.{channel}.{location}.D.{year}.{jday}"
            mseed_path = (
                f"{data_path}/{network}/{year}/{year}.{jday}/{station}.{network}.{channel}.{location}.D.{year}.{jday}"
            )
        elif datacenter.lower() == "scedc":
            location = location.replace("-", "_")
            if "*" not in station:
                station = f"{station:_<5}"
                # mseed_path = f"s3://scedc-pds/continuous_waveforms/{network}/{year}/{year}_{jday}/{network}{station:_<5}{channel}_{location}{year}{jday}.ms"
            mseed_path = f"{data_path}/{year}/{year}_{jday}/{network}{station}{channel}_{location}{year}{jday}.ms"
        else:
            raise ValueError("datacenter not supported")

        fs = fsspec.filesystem(protocol=protocol, anon=anon, token=token)
        print(f"{mseed_path = }")
        mseed_list = sorted(fs.glob(mseed_path))
        if protocol != "file":
            mseed_list = [f"{protocol}://{mseed}" for mseed in mseed_list]

        if group_channel:
            groups = defaultdict(list)
            for mseed in mseed_list:
                if datacenter.lower() == "quakeflow":
                    key = mseed.replace(".mseed", "")[:-1] + "*"
                    groups[key].append(mseed)
                elif datacenter.lower() == "ncedc":
                    key = mseed[:-13] + "*" + mseed[-12:]  # e.g. SQK.BG.DP*..D.2023.079
                    groups[key].append(mseed)
                elif datacenter.lower() == "scedc":
                    key = mseed[:-14] + "*" + mseed[-13:]  # e.g. CISLB__BH*___2015002.ms
                    groups[key].append(mseed)
                else:
                    raise ValueError("datacenter not supported")
            mseed_list = []
            for key in groups:
                mseed_list.append(",".join(sorted(groups[key])))

        return mseed_list

    # %%

    if mseed_list is None:
        # mseed_list = sorted(glob(f"{root_path}/{waveform_dir}/????-???/??/*.mseed"))
        #######
        datacenter = "quakeflow"
        protocol = "file"
        data_path = f"{root_path}/{region}/waveforms"
        network = "??"
        year = "????"
        jday = "???"
        group_channel = True
        #######
        # datacenter = "scedc"
        # protocol = "s3"
        # data_path = "s3://scedc-pds/continuous_waveforms"
        # network = "CI"
        # year = "2023"
        # jday = "001"
        # group_channel = True
        #######
        # datacenter = "ncedc"
        # protocol = "s3"
        # data_path = "s3://ncedc-pds/continuous_waveforms"
        # network = "NC"
        # year = "2023"
        # jday = "001"
        # group_channel = True
        #######
        mseed_list = mapping_mseed_path(
            data_path=data_path,
            network=network,
            year=year,
            jday=jday,
            datacenter=datacenter,
            group_channel=group_channel,
            protocol=protocol,
            anon=True,
            token=token,
        )
        print(f"{len(mseed_list) = }")
        print("\n".join(mseed_list[:5]))
    else:
        with open(f"{root_path}/{region}/{mseed_list}", "r") as fp:
            mseed_list = fp.read().split("\n")

    # %% group channels into stations
    mseed_list = sorted(list(set([x.split(".mseed")[0][:-1] + "*.mseed" for x in mseed_list])))

    # %%
    ## filter out processed events
    # processed_list = sorted(list(fs.glob(f"{result_path}/????-???/??/*.csv")))
    # processed_event_id = [f.name.split("/")[-1].replace(".csv", ".mseed") for f in processed_list]
    # file_list = [f for f in file_list if f.split("/")[-1] not in processed_event_id]
    # mseed_list = sorted(list(set(mseed_list)))

    # %%
    if protocol != "file":
        fs.get(f"{bucket}/{region}/results/data/inventory.xml", f"{root_path}/{region}/results/data/inventory.xml")

    # %%
    with open(f"{root_path}/{result_path}/mseed_list_{rank:03d}.csv", "w") as fp:
        fp.write("fname\n")
        fp.write("\n".join(mseed_list))

    # %%
    os.system(
        f"python {model_path}/phasenet/predict.py --model={model_path}/model/190703-214543 --data_dir=./ --data_list={root_path}/{result_path}/mseed_list_{rank:03d}.csv --response_xml={root_path}/{region}/results/data/inventory.xml --format=mseed --amplitude --highpass_filter=1.0 --result_dir={root_path}/{result_path} --result_fname=phasenet_picks_{rank:03d} --batch_size=1"
    )

    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_picks_{rank:03d}.csv",
            f"{bucket}/{result_path}/phasenet_picks_{rank:03d}.csv",
        )

    # copy to results/phase_picking
    if not os.path.exists(f"{root_path}/{region}/results/phase_picking"):
        os.makedirs(f"{root_path}/{region}/results/phase_picking")
    os.system(
        f"cp {root_path}/{result_path}/phasenet_picks_{rank:03d}.csv {root_path}/{region}/results/phase_picking/phase_picks_{rank:03d}.csv"
    )
    if protocol != "file":
        fs.put(
            f"{root_path}/{result_path}/phasenet_picks_{rank:03d}.csv",
            f"{bucket}/{region}/results/phase_picking/phase_picks_{rank:03d}.csv",
        )

    return f"{result_path}/phasenet_picks_{rank:03d}.csv"


if __name__ == "__main__":
    import json
    import os
    import sys

    root_path = "local"
    region = "demo"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    run_phasenet.execute(root_path=root_path, region=region, config=config)

# %%
