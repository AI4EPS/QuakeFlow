# %%
import json
import os

import fsspec
import pandas as pd
from datetime import timezone, datetime, timedelta
from obspy.clients.fdsn import Client
from obspy import read_inventory
from tqdm import tqdm
import pyproj
import numpy as np
from sklearn.cluster import DBSCAN

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
# %%
root_path = "../../slurm/local"
region = "ca"

# only begin_time, end_time and spatial range are required in this code

# with open(f"{root_path}/{region}/config.json", "r") as fp:
#     config = json.load(fp)
# config["starttime"]= datetime.strptime(config["starttime"]+'.00', "%Y-%m-%dT%H:%M:%S.%f")
# config["endtime"] = datetime.strptime(config["endtime"]+'.00', "%Y-%m-%dT%H:%M:%S.%f")
config = {}

### time bin is 1 day, beacause picks are storaged by day in cloud
### Mendocino Mag 5.4 Event 2023-01-01
config['starttime'] = datetime(2023, 1, 1, 0, 0, 0, 0)
config['endtime'] = datetime(2023, 1, 1, 23, 59, 59, 0)
date_append = '20230101'

# area settings for whole CA
config['minlongitude'] = -128
config['maxlongitude'] = -113
config['minlatitude'] = 30
config['maxlatitude'] = 44
proj = pyproj.Proj(
    f"+proj=sterea +lon_0={(config['minlongitude'] + config['maxlongitude'])/2} +lat_0={(config['minlatitude'] + config['maxlatitude'])/2} +units=km"
)

load_stations = True
load_picks = True
load_catalog = True

providers = ['NC','SC'] 

# %%
if not os.path.exists(f"{root_path}/{region}/obspy"):
    os.makedirs(f"{root_path}/{region}/obspy")
if not os.path.exists(f"{root_path}/{region}/phasenet"):
    os.makedirs(f"{root_path}/{region}/phasenet")
if not os.path.exists(f"{root_path}/{region}/data"):
    os.makedirs(f"{root_path}/{region}/data")
if not os.path.exists(f"{root_path}/{region}/data/stations"):
    os.makedirs(f"{root_path}/{region}/data/stations")
if not os.path.exists(f"{root_path}/{region}/data/picks"):
    os.makedirs(f"{root_path}/{region}/data/picks")
if not os.path.exists(f"{root_path}/{region}/data/picks/picks_csv_list"):
    os.makedirs(f"{root_path}/{region}/data/picks/picks_csv_list")
if not os.path.exists(f"{root_path}/{region}/data/catalog"):
    os.makedirs(f"{root_path}/{region}/data/catalog")

# %%
protocol = "gs"
token_json = f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
with open(token_json, "r") as fp:
    token = json.load(fp)
fs = fsspec.filesystem(protocol, token=token)

# %%
#####################
### load stations ###
#####################
selected_instrument = True
selected_region = True
selected_time = False
if load_stations:
    # stations to csv
    bucket_station = "quakeflow_dataset"
    for provider in providers:
        folder_station = f"{provider}/station"
        station_csv_list = []
        station_csv_list.extend(fs.glob(f"{bucket_station}/{folder_station}/*.csv"))
        ## station storage paths for NC and SC are same
        with open(f"{root_path}/{region}/data/stations/{provider}_station_csv_list.csv", "w") as fp:
            fp.write("\n".join(station_csv_list))
        # with open(f"{root_path}/{region}/data/stations/{provider}_station_csv_list.csv", "r") as fp:
        #     station_csv_list = fp.read().split("\n")

        stations = None
        for station_ in tqdm(station_csv_list, desc=f"loading {provider} stations", total=len(station_csv_list)):
            stations_ = pd.read_csv(
                fs.open(f"{station_}","r"), dtype={"location": str, "station": str, "component": str})
            #     parse_dates=["begin_time", "end_time"],
            #     date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S+00:00")
            # )
            stations = pd.concat([stations, stations_], axis=0, ignore_index=True)

        # some new stations by SCEC don't have end_time
        # I set the empty ones to 3000-01-01T00:00:00+00:00
        stations['end_time'] = stations['end_time'].fillna('3000-01-01T00:00:00+00:00')
        stations['begin_time'] = stations.apply(lambda x: datetime.strptime(x['begin_time'], "%Y-%m-%dT%H:%M:%S+00:00"), axis=1)
        stations['end_time'] = stations.apply(lambda x: datetime.strptime(x['end_time'], "%Y-%m-%dT%H:%M:%S+00:00"), axis=1)

        if selected_instrument:
            channel_types = ['HH','BH','EH','HN', 'DP']
            stations = stations[stations['instrument'].isin(channel_types)]
        if selected_region:
            index = stations['longitude'].between(config["minlongitude"], config["maxlongitude"]) & stations['latitude'].between(config["minlatitude"], config["maxlatitude"])
            stations = stations[index]
        if selected_time:
            stations = stations[(stations["begin_time"] < config["endtime"]) & (stations["end_time"] > config["starttime"])]
        
        stations=stations.reset_index(drop=True)

        stations['location'] = stations['location'].fillna('')
        stations['station_id'] = stations.apply(lambda x: f"{x['network']}.{x['station']}.{x['location']}.{x['instrument']}", axis=1)
        stations['begin_time'] = stations.apply(lambda x: datetime.strftime(x['begin_time'], "%Y-%m-%dT%H:%M:%S.%f"), axis=1)
        stations['end_time'] = stations.apply(lambda x: datetime.strftime(x['end_time'], "%Y-%m-%dT%H:%M:%S.%f"), axis=1)
        stations.to_csv(f"{root_path}/{region}/data/stations/{provider}_stations.csv", index=False)
        stations = stations.drop(columns=['channel'])
        stations_ = stations.groupby("station_id")

        ### Note: for those whose station_id are the same, their time periods and locations might be different
        ### max(stations['longitude'].unique()) - min(stations['longitude'].unique()) ~ 1e-2
        ### SC stations, CE.67910 and CE.13095 have higher residuals (~1km)
        ### NC stations, CE.13095, BK.PKD1, UL.LVFOS have higher residuals (~1km)
        ### Conclusion: one station_id may have multipule location

        # stations to json
        stations_json = {}
        for station_id in stations['station_id'].unique():
            station = stations_.get_group(station_id)
            # print(len(station["sensitivity"].unique()))
            x_km, y_km = proj(station["longitude"].unique()[-1], station["latitude"].unique()[-1]) # use the latest location
            z_km = -station["elevation_m"].unique()[-1] / 1e3
            component = ''.join(station["component"].unique())
            stations_json[station_id] = {
                "network": station["network"].unique()[0],
                "station": station["station"].unique()[0],
                "location": station["location"].unique()[0],
                "instrument": station["instrument"].unique()[0],
                "component": component,
                "sensitivity": [x for x in station['sensitivity']],
                "latitude": station["latitude"].unique()[-1],
                "longitude": station["longitude"].unique()[-1],
                "elevation_m": station["elevation_m"].unique()[-1],
                "depth_km": z_km,
                "x_km": round(x_km, 3),
                "y_km": round(y_km, 3),
                "z_km": round(z_km, 3),
            }
        with open(f"{root_path}/{region}/data/stations/{provider}_stations.json", "w") as fp:
            json.dump(stations_json, fp, indent=4)

    ## merge stations into final csv json file
    stations = []
    for provider in providers:
        stations_ = pd.read_csv(f"{root_path}/{region}/data/stations/{provider}_stations.csv")
        stations_['provider'] = provider
        stations.append(stations_)
    stations = pd.concat(stations, ignore_index=True)
    stations = stations.groupby(["network", "station", "location", "channel"], dropna=False).first().reset_index()
    print(f"Merged {len(stations)} channels")
    stations.to_csv(f"{root_path}/{region}/data/stations/stations.csv", index=False)

    stations = {}
    for provider in providers:
        with open(f"{root_path}/{region}/data/stations/{provider}_stations.json", "r") as fp:
            stations_ = json.load(fp)
        for key, value in stations_.items():
            if key not in stations:
                stations[key] = value
                stations[key]["provider"] = provider
    if len(stations) > 0:
        print(f"Merged {len(stations)} stations")
        with open(f"{root_path}/{region}/data/stations/stations.json", "w") as fp:
            json.dump(stations, fp, indent=4)

# %%
####################
### load catalog ###
####################
#TODO: catalog from NC and SC might need to be merged

# NC catalog is stored by month
# SC catalog is stored by day

# providers = ['NC','SC']
if (config['starttime'] is not None) and (config['endtime'] is not None):
    starttime = config['starttime']
    endtime = config['endtime']
    years = np.arange(starttime.year, endtime.year+1)
    if starttime.year == endtime.year:
        months_of_year = np.arange(starttime.month, endtime.month+1)
        days_of_year = np.arange(starttime.timetuple().tm_yday, endtime.timetuple().tm_yday+1)
    else:
        months_of_year = np.arange(1, 13)
        days_of_year = np.arange(1, 366)
else:
    years = []
    months_of_year = np.arange(1, 13)
    days_of_year = np.arange(1, 366)

def load_catalog_csv(csv_file):
    try:
        with fs.open(csv_file,"r") as f:
            events_temp = pd.read_csv(f)#, parse_dates=['phase_time'])
        return events_temp
    except Exception as e:
        print(f'problem comes from: {csv_file}, the error is: {e}')
        return None

if load_catalog:
    bucket_catalog = "quakeflow_dataset"
    events_ = None
    for provider in providers:
        folder_catalog = f"{provider}/catalog"
        catalog_csv_list = []
        if len(years) == 0:
            if provider == 'NC':
                catalog_csv_list.extend(fs.glob(f"{bucket_catalog}/{folder_catalog}/*.event.csv"))
            if provider == 'SC':
                catalog_csv_list.extend(fs.glob(f"{bucket_catalog}/{folder_catalog}/*/*.event.csv"))
            with open(f"{root_path}/{region}/data/catalog/{provider}_catalog_csv_list.csv", "w") as fp:
                fp.write("\n".join(catalog_csv_list))
        if len(years) > 0:
            for year in years:
                if provider == 'NC':
                    catalog_csv_list.extend(fs.glob(f"{bucket_catalog}/{folder_catalog}/{year}.*.event.csv"))
                if provider == 'SC':
                    catalog_csv_list.extend(fs.glob(f"{bucket_catalog}/{folder_catalog}/{year}/{year}_*.event.csv"))
            with open(f"{root_path}/{region}/data/catalog/{provider}_catalog_csv_list.csv", "w") as fp:
                fp.write("\n".join(catalog_csv_list))

        if provider == 'NC':
            catalog_csv_info = pd.DataFrame(catalog_csv_list, columns=["csv"])
            catalog_csv_info['year'] = catalog_csv_info.apply(lambda x: int(x['csv'].split("/")[-1].split(".")[0]), axis=1)
            catalog_csv_info['month_of_year'] = catalog_csv_info.apply(lambda x: int(x['csv'].split("/")[-1].split(".")[1]), axis=1)
            index = catalog_csv_info['year'].isin(years) & catalog_csv_info['month_of_year'].isin(months_of_year)
            catalog_csv_list = catalog_csv_info[index]['csv'].to_list()
        if provider == 'SC':
            catalog_csv_info = pd.DataFrame(catalog_csv_list, columns=["csv"])
            catalog_csv_info['year'] = catalog_csv_info.apply(lambda x: int(x['csv'].split("/")[-1].split("_")[0]), axis=1)
            catalog_csv_info['day_of_year'] = catalog_csv_info.apply(lambda x: int(x['csv'].split("/")[-1].split("_")[1].split(".")[0]), axis=1)
            index = catalog_csv_info['year'].isin(years) & catalog_csv_info['day_of_year'].isin(days_of_year)
            catalog_csv_list = catalog_csv_info[index]['csv'].to_list()

        events = None
        # for event_csv in tqdm(catalog_csv_list, desc=f"loading {provider} catalog", total=len(catalog_csv_list)):
        #     events_ = pd.read_csv(
        #         fs.open(f"{event_csv}","r"),
        #     )
        #     events = pd.concat([events, events_], axis=0, ignore_index=True)
        with ThreadPoolExecutor() as executor:
            events_list = list(tqdm(executor.map(load_catalog_csv, catalog_csv_list), total=len(catalog_csv_list), desc=f"loading {provider}_catalog"))

        events = pd.concat([p for p in events_list if p is not None], ignore_index=True)
        events['time'] = events.apply(lambda x: datetime.strptime(x['time'], "%Y-%m-%dT%H:%M:%S.%f+00:00"), axis=1)
        index = events['time'].between(config['starttime'], config['endtime'])
        events = events[index]
        events['time'] = events.apply(lambda x: datetime.strftime(x['time'], "%Y-%m-%dT%H:%M:%S.%f"), axis=1)
        events_ = pd.concat([events_, events], ignore_index=True)
        events.to_csv(f"{root_path}/{region}/data/catalog/{provider}_events.csv", index=False)
    events_.to_csv(f"{root_path}/{region}/data/catalog/events.csv", index=False)
# %%
##################
### load picks ###
##################
save_empty_files = False
networks = []
if (config['starttime'] is not None) and (config['endtime'] is not None):
    starttime = config['starttime']
    endtime = config['endtime']
    years = np.arange(starttime.year, endtime.year+1)
    if starttime.year == endtime.year:
        days_of_year = np.arange(starttime.timetuple().tm_yday, endtime.timetuple().tm_yday+1)
    else:
        days_of_year = np.arange(1, 366)
else:
    years = []
    days_of_year = np.arange(1, 366)
days_of_year = [str(x).rjust(3, '0') for x in days_of_year]

# %%
if load_picks:
    picks_ = None
    bucket_phasenet = "quakeflow_catalog"
    for provider in providers:
        folder_phasenet = f"{provider}/phasenet"

        # TODO: CE network seems to be not fully processed
        if provider == 'NC':
            if len(networks) == 0:
                networks_list = fs.ls(f"{bucket_phasenet}/{folder_phasenet}")
                for network in networks_list:
                    if (network.split("/")[-1] != "mseed_list") and (network.split("/")[-1] != "csv_list"):
                        networks.append(network.split("/")[-1])
            csv_list = []
            for network in networks:
                if len(years) == 0:
                    years_list = fs.ls(f"{bucket_phasenet}/{folder_phasenet}/{network}")
                    for year in years_list:
                        years.append(year.split("/")[-1])
                csv_list_ = []
                for year in years:
                    for day_of_year in days_of_year:
                        csv_list_.extend(fs.glob(f"{bucket_phasenet}/{folder_phasenet}/{network}/{year}/{year}.{day_of_year}/*.csv"))
                with open(f"{root_path}/{region}/data/picks/picks_csv_list/{provider}_{network}_csv_list.txt", "w") as fp:
                    fp.write("\n".join(csv_list_))
                if len(csv_list_) > 0 and csv_list_[0] != '':
                    csv_list.extend(csv_list_)
            with open(f"{root_path}/{region}/data/picks/picks_csv_list/{provider}_csv_list.txt", "w") as fp:
                fp.write("\n".join(csv_list))
        
        if provider == 'SC':
            csv_list = []
            for year in years:
                for day_of_year in days_of_year:
                    csv_list.extend(fs.glob(f"{bucket_phasenet}/{folder_phasenet}/{year}/{year}_{day_of_year}/*.csv"))
            with open(f"{root_path}/{region}/data/picks/picks_csv_list/{provider}_csv_list.txt", "w") as fp:
                fp.write("\n".join(csv_list))

        if provider == 'NC':
            csv_list = pd.DataFrame(csv_list, columns=["csv"])
            csv_list['time'] = csv_list.apply(lambda x: x['csv'].split("/")[5], axis=1)
            csv_list['year'] = csv_list.apply(lambda x: int(x['time'].split(".")[0]), axis=1)
            csv_list['day_of_year'] = csv_list.apply(lambda x: int(x['time'].split(".")[1]), axis=1)
            csv_list['station'] = csv_list.apply(lambda x: x['csv'].split("/")[-1].split('.')[0], axis=1)
            csv_list['network'] = csv_list.apply(lambda x: x['csv'].split("/")[-1].split('.')[1], axis=1)
            csv_list['channel'] = csv_list.apply(lambda x: x['csv'].split("/")[-1].split('.')[2][:2], axis=1)
            csv_list['location'] = csv_list.apply(lambda x: x['csv'].split("/")[-1].split('.')[3], axis=1)
        if provider == 'SC':
            csv_list = pd.DataFrame(csv_list, columns=["csv"])
            csv_list['time'] = csv_list.apply(lambda x: x['csv'].split("/")[4], axis=1)
            csv_list['year'] = csv_list.apply(lambda x: int(x['time'].split("_")[0]), axis=1)
            csv_list['day_of_year'] = csv_list.apply(lambda x: int(x['time'].split("_")[1]), axis=1)
            csv_list['network'] = csv_list.apply(lambda x: x['csv'].split("/")[-1][:2], axis=1)
            csv_list['station'] = csv_list.apply(lambda x: x['csv'].split("/")[-1][2:7].replace('_', ''), axis=1)
            csv_list['channel'] = csv_list.apply(lambda x: x['csv'].split("/")[-1][7:9], axis=1)
            csv_list['location'] = csv_list.apply(lambda x: x['csv'].split("/")[-1][10:13].replace('_', ''), axis=1)

        print(f'unique channels of {provider} picks: {csv_list["channel"].unique()}')

        ## merge picks by channel priority
        filter_picks_by_channel = False
        if filter_picks_by_channel:
            channel_priority = {'HH':1, 'BH':2, 'EH':3, 'HN':4, 'DP':5}
            csv_list['priority'] = csv_list['channel'].map(channel_priority)
            csv_list = csv_list.sort_values(by=['priority']).groupby(['network', 'station', 'time']).first().reset_index()
            csv_list = csv_list.drop(columns=['priority'])

        ## un-paralleled version
        ## slow when reading 1year data! (~ 40hour for 1-year picks)
        # picks = None
        # for i, row in tqdm(csv_list.iterrows(), desc="loading picks", total = len(csv_list)):
        #     try:
        #         picks_temp = pd.read_csv(fs.open(row['csv'], "r"), parse_dates=['phase_time'])
        #         if picks is None:
        #             picks = picks_temp
        #         else:
        #             picks = pd.concat([picks, picks_temp], ignore_index=True)
        #     except Exception as e:
        #         print(f'problem comes from: {row["csv"]}, the error is: {e}')

        empty_files =[]
        def load_csv(csv_file):
            # print(csv_file)
            if save_empty_files:
                global empty_files
            try:
                with fs.open(csv_file,"r") as f:
                    picks_temp = pd.read_csv(f, parse_dates=['phase_time'], dtype={"phase_amplitude": str})
                return picks_temp
            except Exception as e:
                # print(f'problem comes from: {csv_file}, the error is: {e}')
                if save_empty_files:
                    empty_files.append(csv_file)
                return None

        with ThreadPoolExecutor() as executor:
            picks_list = list(tqdm(executor.map(load_csv, csv_list['csv']), total=len(csv_list), desc=f"loading {provider} picks"))

        picks = pd.concat([p for p in picks_list if p is not None], ignore_index=True)
        index = picks['phase_time'].between(config['starttime'], config['endtime'])
        picks = picks[index]
        picks['phase_time'] = picks.apply(lambda x: datetime.strftime(x['phase_time'], "%Y-%m-%dT%H:%M:%S.%f"), axis=1)
        picks['providing_source'] = provider
        picks.to_csv(f"{root_path}/{region}/data/picks/{provider}_picks.csv", index=False)
        picks_ = pd.concat([picks_, picks], ignore_index=True)
        if save_empty_files:
            empty_picks = pd.DataFrame(empty_files, columns=["file_path"])
            empty_picks['year'] = empty_picks.apply(lambda x: x['file_path'].split("/")[4], axis=1)
            empty_picks['day_of_year'] = empty_picks.apply(lambda x: x['file_path'].split("/")[5].split(".")[1], axis=1)
            empty_picks['network'] = empty_picks.apply(lambda x: x['file_path'].split("/")[6].split(".")[1], axis=1)
            empty_picks['station'] = empty_picks.apply(lambda x: x['file_path'].split("/")[6].split(".")[0], axis=1)
            empty_picks['channel'] = empty_picks.apply(lambda x: x['file_path'].split("/")[6].split(".")[2], axis=1)
            empty_picks['location'] = empty_picks.apply(lambda x: x['file_path'].split("/")[6].split(".")[3], axis=1)
            empty_picks.to_csv(f"{root_path}/{region}/data/picks/{provider}_empty_picks.csv", index=False)
    picks_.to_csv(f"{root_path}/{region}/data/picks/picks.csv", index=False)
# %%
#################
## Merge picks ##
#################
merge_picks_by_cluster = True # highly recommended
if merge_picks_by_cluster:
    MIN_DISTANCE = 1e-3 # minimum distance to sperate stations into groups
    MIN_SECOND = 1 # minimum seconds to merge picks

    stations = pd.read_json(f"{root_path}/{region}/data/stations/stations.json", orient="index")
    stations['station_id'] = stations.index
    stations.reset_index(drop=True, inplace=True)
    stations['id'] = stations.apply(lambda x: f"{x['network']}.{x['station']}", axis=1)

    # Make sure different instruments of the same station are grouped together
    stations['longitude_'] = stations['longitude']
    stations['latitude_'] = stations['latitude']
    for id, group in stations.groupby('id'):
        mean_lon = group['longitude'].mean()
        mean_lat = group['latitude'].mean()
        idx = group.index
        stations.loc[idx, 'longitude_'] = mean_lon
        stations.loc[idx, 'latitude_'] = mean_lat

    # generate station mapping
    station_mapping = {}
    data = stations[['longitude_','latitude_']].values
    db = DBSCAN(eps= MIN_DISTANCE, min_samples=1).fit(data)
    labels = db.labels_
    unique_labels = set(labels)
    unique_labels = unique_labels.difference([-1])
    stations['label'] = labels
    for label in unique_labels:
        station = stations[stations['label']==label].copy()
        for id in station['station_id'].unique():
            station_mapping[id] = station['label'].unique()[0]
    stations.drop(columns=['longitude_', 'latitude_', 'label'], inplace=True)
    
    picks_filt =  pd.read_csv(f"{root_path}/{region}/data/picks/picks.csv", dtype={"phase_amplitude": str}) # keep phase_amplitude as string to avoid float precision lost
    picks_filt = picks_filt.merge(stations[['station_id','longitude', 'latitude']], on='station_id')
    picks_filt = picks_filt[picks_filt['station_id'].isin(stations['station_id'].unique())]
    picks_filt['location_label'] = picks_filt['station_id'].map(lambda x: station_mapping[x] if x in station_mapping else None)
    picks_filt['timestamp'] = picks_filt.apply(lambda x: datetime.strptime(x['phase_time'], "%Y-%m-%dT%H:%M:%S.%f"), axis=1)
    picks_filt['t'] =( picks_filt['timestamp'] - picks_filt['timestamp'].min()).dt.total_seconds()
    picks_filt['selection'] = False

    ## Unparallel version
    # for (id, phase_type), group in tqdm(picks_filt.groupby(["location_label","phase_type"])):
    #     db = DBSCAN(eps=MIN_SECOND, min_samples=1).fit(group[["t"]])
    #     labels = db.labels_
    #     group['label'] = labels
    #     idx = group.groupby('label')['phase_score'].transform(lambda x: x.idxmax())
    #     picks_filt.loc[idx, 'selection'] = True
    # picks_filt = picks_filt[picks_filt['selection']].copy()

    # parallel version
    def process_group(group_tuple):
        _, group = group_tuple
        db = DBSCAN(eps=MIN_SECOND, min_samples=1).fit(group[["t"]])
        labels = db.labels_
        group = group.copy()  # Make a copy to avoid modifying the original dataframe
        group['label'] = labels
        idx_max = group.groupby('label')['phase_score'].idxmax().values
        return idx_max

    def process_in_batches(groups, batch_size):
        with tqdm(total=len(groups), desc="Merging picks") as pbar:
            for i in range(0, len(groups), batch_size):
                batch = groups[i:i+batch_size]
                with ProcessPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(process_group, group) for group in batch]
                    for future in as_completed(futures):
                        try:
                            idx_max = future.result()
                            if len(idx_max) > 0:
                                picks_filt.loc[idx_max, 'selection'] = True
                        except Exception as e:
                            print(f"Error processing batch: {e}")
                        finally:
                            # Updating the progress bar after each group is processed
                            pbar.update(1)

    groups = [(name, group) for name, group in picks_filt.groupby(["location_label", "phase_type"])]
    batch_size = 32
    process_in_batches(groups, batch_size)
    picks_filt = picks_filt[picks_filt['selection']].copy()
    picks_filt.drop(columns=['longitude', 'latitude', 'location_label', 'timestamp', 't', 'selection'], inplace=True)
    picks_filt.reset_index(drop=True, inplace=True)
    picks_filt.to_csv(f"{root_path}/{region}/data/picks/picks.csv", index=False)

# %%
if (load_picks) and (os.path.exists(f"{root_path}/{region}/data/picks/picks.csv")):
    os.system(f"cp {root_path}/{region}/data/picks/picks.csv {root_path}/{region}/phasenet/phasenet_picks.csv")

if (load_stations) and (os.path.exists(f"{root_path}/{region}/data/stations/stations.csv")):
    os.system(f"cp {root_path}/{region}/data/stations/stations.json {root_path}/{region}/obspy/stations.json")
    os.system(f"cp {root_path}/{region}/data/stations/stations.csv {root_path}/{region}/obspy/stations.csv")

if (load_catalog) and (os.path.exists(f"{root_path}/{region}/data/catalog/events.csv")):
    os.system(f"cp {root_path}/{region}/data/catalog/events.csv {root_path}/{region}/obspy/events.csv")
# %%
