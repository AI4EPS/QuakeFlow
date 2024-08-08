# %%
### This code is used to convert velest output into standard format.
### This code can reserve the event index, although running velest will lose event index.
# %%
import numpy as np
import obspy
# import re
import json
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from tqdm import tqdm
from pyproj import Proj

# region = "Mendocino_on_off"
# region = "Mendocino_onshore_offshore"
root_path="local"
# root_path="nfs"
region = "Mendocino_8mon"
# region = "Mendocino_3year"
# region = "Mendocino_15day"

data_path = f"{region}/velest"
gamma_path = f"{region}/gamma"


# %%
def parse_time(time):
    year = '20' + time[:2]

    # month_delta = 0
    if int(time[2:4]) == 0:
        month = '1'
    else:
        month = str(int(time[2:4]))
    
    if int(time[4:6]) == 0:
        day = '01'
    else:
        day = str(int(time[4:6])).rjust(2,'0')
        
    hour = str(int(time[7:9])).rjust(2,'0')

    hour_delta = 0
    if int(time[9:11]) == 60.0:
        hour_delta = 1
        minute = '00'
    else:
        minute = str(int(time[9:11])).rjust(2,'0')
    
    minute_delta = 0
    if float(time[12:18]) == 60.0:
        minute_delta = 1
        second = '0.00'
    else:
        second = str(float(time[12:18]))
    event_time = year+'-'+month+'-'+day+'T'+hour+':'+minute+':'+second
    time_obj = datetime.strptime(event_time, "%Y-%m-%dT%H:%M:%S.%f") + relativedelta(hours=hour_delta, minutes=minute_delta)
    return time_obj
# %%
station_json = f"{region}/obspy/stations.json"
stations = pd.read_json(f"{root_path}/{station_json}", orient="index")

stations['station_id'] = stations.index.values
stations = stations.reset_index()
stations["index"] = 'ST' + stations.index.astype(str)
# %%
with open(f"{root_path}/{region}/config.json","r") as fp:
    config = json.load(fp)
with open(f"{root_path}/{data_path}/single_out.CHECK", 'r') as file2:
    lines2 = file2.readlines()
with open(f"{root_path}/{data_path}/velest.CNV", 'r') as file:
    lines = file.readlines()
picks_csv = f"{root_path}/{gamma_path}/gamma_picks.csv"
catalog_csv = f"{root_path}/{gamma_path}/gamma_events.csv"
picks = pd.read_csv(picks_csv)
events = pd.read_csv(catalog_csv)

if "longitude0" not in config:
    config["longitude0"] = (config["minlongitude"] + config["maxlongitude"]) / 2
if "latitude0" not in config:
    config["latitude0"] = (config["minlatitude"] + config["maxlatitude"]) / 2
proj = Proj(f"+proj=sterea +lon_0={config['longitude0']} +lat_0={config['latitude0']} +units=km")

# %%
# remove unlocated events
velest_index = np.zeros(len(events))
index = 0
for line2 in tqdm(lines2, desc="Removing Unlocated Events", total=len(lines2)):
    if line2[:11] == '1 E V E N T':
        index = index + 1
    if line2[:6] == '0 DATE':
        velest_index[index-1] = 1
# %%
events['velest_locate'] = velest_index
events_located = events[events['velest_locate'] != 0].copy()
events_located.reset_index(drop = True, inplace = True)
# picks_located = picks.copy()
# picks_by_event = picks_located.groupby("event_index").groups

gap = np.zeros(len(events_located))
rms = np.zeros(len(events_located))
event_id = -1
for line in tqdm(lines, desc = "Converting Velest to Standard format", total=len(lines)):
    if line[:2] != 'ST' and line[:2] != '\n':
        event_id = event_id+1
        rms[event_id] = float(line[63:67])
        gap[event_id] = int(line[54:57])
        time = line[:18]
        line = line[18:]
        time_obj = parse_time(time)
        event_time = time_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")
        event_info = line.split()
        # print(event_info)
        if event_info[0][-1] == 'N':
            event_lat = float(event_info[0][:-1])
        else:
            event_lat = -1 * float(event_info[0][:-1])
        if event_info[1][-1] == 'S':
            event_lon = float(event_info[1][:-1])
        else:
            event_lon = -1 * float(event_info[1][:-1])
        events_located.at[int(event_id),'latitude'] = event_lat
        events_located.at[int(event_id),'longitude'] = event_lon
        events_located.at[int(event_id),'depth_km'] = float(event_info[2])
        events_located.at[int(event_id),'time'] = event_time
        event_index = events_located.at[int(event_id), 'event_index']
    ## Because signle mode is used in Velest, only earthquake times and locations are modified
    ## The phase times are unchanged, but the travel times are updated based on changes of event time correspondingly.
    ## So it's not needed to update the picks file 
    #  Update picks
    # if line[:2] == 'ST':
    #     station_info = line.split('ST')
    #     station_info = station_info[1:]
    #     for pick in station_info:
    #         if pick[4:6] == 'P0':
    #             phase_type = 'P'
    #         if pick[4:6] == 'S0':
    #             phase_type = 'S'
    #         station_id = stations[stations['index'] == 'ST'+str(int(pick[:4]))]['station_id'].values[0]
    #         phase_time_obj = time_obj + timedelta(seconds = float(pick[6:12]))
    #         phase_time = phase_time_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")
    #         picks_located.loc[(picks_located['station_id'] == station_id) & 
    #                           (picks_located['phase_type'] == phase_type) & 
    #                           (picks_located['event_index'] == event_index), 'phase_time'] = phase_time   
           
# %%
events_located[["x(km)", "y(km)"]] = events_located.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
events_located["z(km)"] = events_located["depth_km"]


events_located['rms'] = rms
events_located['gap'] = gap
events_located.to_csv(f'{root_path}/{data_path}/velest_unfilter_events.csv', index=False)

# fill out bad events by rms, gap and depth
# parameters in loc flow:
# rms_threshold = 0.5 sec, gap_threshold = 300 in degree
rms_threshold = 1
gap_threshold = 400
depth_max = 65

events_located = events_located[(events_located['rms'] < rms_threshold) & (events_located['gap'] < gap_threshold) & (events_located['depth_km'] < 65)]
events_located.drop('velest_locate', axis=1, inplace=True)
# events_located.drop('rms', axis=1, inplace=True)
# events_located.drop('gap', axis=1, inplace=True)

# %%
events_located.to_csv(f'{root_path}/{data_path}/velest_events.csv', index=False)
picks.to_csv(f'{root_path}/{data_path}/velest_picks.csv', index=False)
# %%
