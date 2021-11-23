#%%
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
stations = pd.read_csv('stations.csv', sep="\t")

# %%
converted_hypoinverse = []
converted_hypoDD = {}

for i in tqdm(range(len(stations))):

    network_code, station_code, comp_code, channel_code = stations.iloc[i]['station'].split('.')
    station_weight = " "
    lat_degree = int(stations.iloc[i]['latitude'])
    lat_minute = (stations.iloc[i]['latitude'] - lat_degree) * 60
    north = "N" if lat_degree >= 0 else "S"
    lng_degree = int(stations.iloc[i]['longitude'])
    lng_minute = (stations.iloc[i]['longitude'] - lng_degree) * 60
    west = "W" if lng_degree <= 0 else "E"
    elevation = stations.iloc[i]['elevation(m)']
    line_hypoinverse = f"{station_code:<5} {network_code:<2} {comp_code[:-1]:<1}{channel_code:<3} {station_weight}{abs(lat_degree):2.0f} {abs(lat_minute):7.4f}{north}{abs(lng_degree):3.0f} {abs(lng_minute):7.4f}{west}{elevation:4.0f}\n"
    # line_hypoDD = f"{network_code:<2}.{station_code:<5} {stations.iloc[i]['latitude']:.3f}, {stations.iloc[i]['longitude']:.3f}\n"
    #line_hypoDD = f"{station_code} {stations.iloc[i]['latitude']:.3f} {stations.iloc[i]['longitude']:.3f}\n"
    converted_hypoinverse.append(line_hypoinverse)
    #converted_hypoDD.append(line_hypoDD)
    converted_hypoDD[f"{station_code}"] = f"{station_code} {stations.iloc[i]['latitude']:.3f} {stations.iloc[i]['longitude']:.3f}\n"

# %%
out_file = 'stations_hypoinverse.dat'
with open(out_file, 'w') as f:
    f.writelines(converted_hypoinverse)

out_file = 'stations_hypoDD.dat'
# converted_hypoDD = list(set(converted_hypoDD))
with open(out_file, 'w') as f:
    #f.writelines(converted_hypoDD)
    for k, v in converted_hypoDD.items():
        f.write(v)

# %%
