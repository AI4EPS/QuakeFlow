# %%
import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

#region = "Mendocino_test"
#region = "Mendocino_on_off"
# region = "Mendocino_onshore_offshore"
root_path = "local"
# root_path = "nfs"
region = "Mendocino_8mon"
# region = "Mendocino_3year"
# region = "Mendocino_15day"

data_path = f"{region}/gamma"
result_path = f"{region}/velest"

if not os.path.exists(f"{root_path}/{result_path}"):
    os.makedirs(f"{root_path}/{result_path}")
# %%
def station_format(row):
    return f"{str(row['index']):6s}{row['LAT']} {row['LON']} {-1*row['elevation_m']:4.0f} 1 {str(row['index'][2:]).rjust(3,'0')} {row['pdelay']:5.2f}  {row['sdelay']:5.2f}   {row['imod']:1.0f}\n"
# %%
## velest parameter control file (*.cmn)
with open(f"{root_path}/{region}/config.json", "r") as f:
    config = json.load(f)
print(json.dumps(config, indent=4, sort_keys=True))

if "center" not in config:
    config["center"] = [(config["minlongitude"] + config["maxlongitude"]) / 2, 
                        (config["minlatitude"] + config["maxlatitude"]) / 2]
    config["xlim_degree"] = [config["minlongitude"], config["maxlongitude"]]
    config["ylim_degree"] = [config["minlatitude"], config["maxlatitude"]]

# %%
## write station file (*.sta)
station_json = f"{region}/obspy/stations.json"
stations = pd.read_json(f"{root_path}/{station_json}", orient="index")

# %%
stations['station_id'] = stations.index.values
stations = stations.reset_index()
stations["index"] = 'ST' + stations.index.astype(str)
stations["LAT"] = stations.apply(lambda stations: str(f"{stations['latitude']:7.4f}")+'N' if stations["latitude"] > 0 else str(f"{-1*stations['latitude']:7.4f}")+'S', axis = 1)
stations["LON"] = stations.apply(lambda stations: str(f"{stations['longitude']:8.4f}")+'E' if stations["longitude"] > 0 else str(f"{-1*stations['longitude']:8.4f}")+'W', axis = 1)
stations["pdelay"] = [0.00] * len(stations['index'])
stations["sdelay"] = [0.00] * len(stations['index'])
stations["imod"] = [1] * len(stations['index'])

stations["sta2velest"] = stations.apply(station_format, axis=1)
f = open(f'{root_path}/{result_path}/station.sta', 'w')
f.write('(a6,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)\n')
f.writelines(stations['sta2velest'])
f.write('\n')
f.close()

shift_topo = stations["elevation_m"].max()/1e3
# %%
## write earthquake file (*.cnv)
picks_csv = f"{root_path}/{data_path}/gamma_picks.csv"
catalog_csv = f"{root_path}/{data_path}/gamma_events.csv"
picks = pd.read_csv(picks_csv)
events = pd.read_csv(catalog_csv)

# %%
events.sort_values("time", inplace=True)
picks = picks.loc[picks["event_index"].isin(events["event_index"])]

events["LAT"] = events.apply(lambda events: str(f"{events['latitude']:7.4f}")+'N' if events["latitude"] > 0 else str(f"{-1*events['latitude']:7.4f}")+'S', axis = 1)
events["LON"] = events.apply(lambda events: str(f"{events['longitude']:8.4f}")+'E' if events["longitude"] > 0 else str(f"{-1*events['longitude']:8.4f}")+'W', axis = 1)

lines = []
picks_by_event = picks.groupby("event_index").groups
for i, event in tqdm(events.iterrows(), desc="Convert gamma catalog to velest", total = len(events)):
    if i > 0:
        lines.append('\n')
    event_time = datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S.%f")
    lat = event["LAT"]
    lng = event["LON"]
    dep = event["depth_km"] + shift_topo
    mag = event["magnitude"]
    # EH = 0
    # EZ = 0
    # RMS = event["sigma_time"]

    year, month, day, hour, min, sec = (
        event_time.year,
        event_time.month,
        event_time.day,
        event_time.hour,
        event_time.minute,
        float(event_time.strftime("%S.%f")),
    )
    event_line = f"{str(year)[2:]}{str(month).rjust(2,' ')}{str(day).rjust(2,' ')} {str(hour).rjust(2,' ')}{str(min).rjust(2,' ')} {sec:5.2f} {lat} {lng} {dep:8.4f} {mag:7.2f} 0\n"
    lines.append(event_line)

    picks_idx = picks_by_event[event["event_index"]]
    count = 0
    for j in picks_idx:
        count += 1
        pick =picks.loc[j]
        sta_id = stations[stations['station_id'] == pick['station_id']]['index'].values[0]
        phase_type = pick["phase_type"].upper()
        phase_score = pick["phase_score"]
        pick_time = (datetime.strptime(pick["phase_time"], "%Y-%m-%dT%H:%M:%S.%f") - event_time).total_seconds()
        pick_line = f"{str(sta_id).ljust(6,' ')}{phase_type}0{pick_time:6.2f}" 
        lines.append(pick_line)
        if count == 6:
            count = 0
            lines.append('\n')
    if count != 0:
        lines.append('\n')  
    # lines.append('\n')
# lines.append("\n")
with open(f'{root_path}/{result_path}/catalog.cnv', 'w') as f:
    f.writelines(lines)
f.close()

with open(f'{root_path}/{result_path}/center.txt', 'w') as f:
    f.write(str(f"{config['center'][1]:8.4f}")+' '+str(f"{-1*config['center'][0]:8.4f}")+' '+str(len(events['event_index'])).rjust(3,'0'))
f.close()
    


# %%
