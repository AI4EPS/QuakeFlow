# %%
import os
from datetime import datetime

import pandas as pd
from args import parse_args
from tqdm import tqdm

args = parse_args()

# %%
root_path = args.root_path
region = args.region
result_path = f"{region}/growclust"
if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")

# %%
# stations_json = f"{region}/results/data/stations.json"
# stations = pd.read_json(f"{root_path}/{stations_json}", orient="index")
station_csv = f"{region}/cctorch/cctorch_stations.csv"
stations = pd.read_csv(f"{root_path}/{station_csv}")
stations.set_index("station_id", inplace=True)


lines = []
for i, row in stations.iterrows():
    # line = f"{row['network']}{row['station']:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    line = f"{row['station']:<4} {row['latitude']:.4f} {row['longitude']:.4f}\n"
    lines.append(line)

with open(f"{root_path}/{result_path}/stlist.txt", "w") as fp:
    fp.writelines(lines)


# %%
# events_csv = f"{region}/results/phase_association/events.csv"
# events_csv = f"{region}/adloc/ransac_events.csv"
events_csv = f"{region}/cctorch/cctorch_events.csv"
# event_file = f"{region}/cctorch/events.csv"
events = pd.read_csv(f"{root_path}/{events_csv}")
# event_df = event_df[event_df["gamma_score"] > 10]
# event_index = [f"{x:06d}" for x in event_df["event_index"]]
# events["time"] = pd.to_datetime(events["time"])
events["time"] = pd.to_datetime(events["event_time"])
if "magnitude" not in events.columns:
    events["magnitude"] = 0.0

events[["year", "month", "day", "hour", "minute", "second"]] = (
    events["time"]
    # .apply(lambda x: datetime.fromisoformat(x).strftime("%Y %m %d %H %M %S.%f").split(" "))
    .apply(lambda x: x.strftime("%Y %m %d %H %M %S.%f").split(" "))
    .apply(pd.Series)
    .apply(pd.to_numeric)
)

lines = []
for i, row in events.iterrows():
    # yr mon day hr min sec lat lon dep mag eh ez rms evid
    line = f"{row['year']:4d} {row['month']:2d} {row['day']:2d} {row['hour']:2d} {row['minute']:2d} {row['second']:7.3f} {row['latitude']:.4f} {row['longitude']:.4f} {row['depth_km']:7.3f} {row['magnitude']:.2f} 0.000 0.000 0.000 {row['event_index']:6d}\n"
    lines.append(line)

with open(f"{root_path}/{result_path}/evlist.txt", "w") as fp:
    fp.writelines(lines)

# %%
os.system(f"bash run_growclust_cc.sh {root_path} {region}")
