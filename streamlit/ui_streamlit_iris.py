import streamlit as st
from collections import defaultdict, deque
from kafka import KafkaConsumer
from json import loads
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import PIL
from PIL import Image
import streamlit.components.v1 as components
import os
import tweepy
import logging
import sys
from collections import deque
from geopy.geocoders import Nominatim
import threading
import pickle


# # Streamlit layout CSS
# st.markdown(
#     f"""
# <style>
#     .reportview-container .main .block-container{{
#         max-width: 100vw;
#         padding-top: 1rem;
#         padding-right: 1rem;
#         padding-left: 1rem;
#         padding-bottom: 1rem;
#     }}
#     .reportview-container .main {{
#         color: black;
#         background-color: white;
#     }}
# </style>
# """,
#     unsafe_allow_html=True,
# )

# # Lambdas and Constants
normalize =lambda x: (x - np.mean(x) + np.finfo(x.dtype).eps) / (np.std(x) + np.finfo(x.dtype).eps)
timestamp_seconds =lambda x: datetime.fromisoformat(x).timestamp()

# EVENT_MIN_GAP = 5
# WINDOW_LENGTH = 100
# WINDOW_NUMBER = 60
# HOP_LENGTH = 10
# REFRESH_SEC = 1.0
# MAP_WIDTH = 900
# MAP_HEIGHT = 650
# MAP_ZOOM = 9
# BOT_MAGNITUDE_THRESHOLD = 1.5
# GEOLOC_TOUT = 5  # in seconds
# I_MADE_A_TWEET = False
# dt = 0.01
# prev_event_bundle = None
# prev_event_bundle = (0.0, 0.0, 0.0, 0.0)

CONFIG_PKL = "config_hawaii.pkl"
STATION_CSV = "stations_hawaii.csv"
with open(CONFIG_PKL, "rb") as fp:
    CONFIG = pickle.load(fp)
STATIONS = pd.read_csv(STATION_CSV, delimiter="\t")
STATIONS = STATIONS.rename(columns={"station":"id"})

consumer = None
# Connection to Kafka
try:
    print('Connecting to k8s kafka')
    BROKER_URL = 'quakeflow-kafka:9092'
    consumer = KafkaConsumer(
        bootstrap_servers=[BROKER_URL],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        key_deserializer=lambda x: loads(x.decode('utf-8')),
        value_deserializer=lambda x: loads(x.decode('utf-8'))
    )
    use_kafka = True
    print('k8s kafka connection success!')
    consumer.subscribe(['waveform_raw', 'phasenet_picks', 'gmma_events'])
except BaseException:
    print('k8s Kafka connection error')
    try:
        print('Connecting to local kafka')
        BROKER_URL = 'localhost:9092'
        consumer = KafkaConsumer(
            bootstrap_servers=[BROKER_URL],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            key_deserializer=lambda x: loads(x.decode('utf-8')),
            value_deserializer=lambda x: loads(x.decode('utf-8'))
        )
        use_kafka = True
        print('local kafka connection success!')
        consumer.subscribe(['waveform_raw', 'phasenet_picks', 'gmma_events'])

    except BaseException:
        print('local Kafka connection error')

if not consumer:
    raise('No kafka server found!')

# # Setting up Tweepy
# consumer_key = os.getenv('CONSUMER_KEY')
# consumer_secret = os.getenv('CONSUMER_SECRET')
# access_token = os.getenv('ACCESS_TOKEN')
# access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')
# print(consumer_key)
# print(consumer_secret)
# print(access_token)
# print(access_token_secret)

# logger = logging.getLogger()

# def create_api():
#     consumer_key = os.getenv("CONSUMER_KEY")
#     consumer_secret = os.getenv("CONSUMER_SECRET")
#     access_token = os.getenv("ACCESS_TOKEN")
#     access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

#     if not consumer_key:
#         return
#     try:
#         auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#         auth.set_access_token(access_token, access_token_secret)
#         api = tweepy.API(auth, wait_on_rate_limit=True,
#                          wait_on_rate_limit_notify=True)
#         api.verify_credentials()
#         logger.info("API created")
#         return api
#     except Exception as e:
#         logger.error("Error creating API", exc_info=True)
#     return None

# api = create_api()

# # Functions

# def latlon2address(lat, lon, geolocator):
#     try:
#         location = geolocator.reverse(f"{lat}, {lon}")
#         print(location)
#         return location.address
#     except BaseException:
#         return None

# geolocator = Nominatim(user_agent="https", timeout=5)

# def update_figure_layout(figure):
#     figure.update_layout(
#         mapbox_style="white-bg",
#         mapbox_layers=[
#             {
#                 "below": 'traces',
#                 "sourcetype": "raster",
#                 "sourceattribution": "United States Geological Survey",
#                 "source": [
#                     "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
#                 ]
#             }
#         ])
#     figure.update_layout(
#         showlegend=True,
#         width=MAP_WIDTH,
#         height=MAP_HEIGHT,
#         geo=dict(
#             landcolor='rgb(217, 217, 217)',
#             lonaxis=dict(
#                 showgrid=True,
#                 gridwidth=0.05,
#                 range=CONFIG["xlim_degree"],
#                 dtick=5
#             ),
#             lataxis=dict(
#                 showgrid=True,
#                 gridwidth=0.05,
#                 range=CONFIG["ylim_degree"],
#                 dtick=5
#             )
#         ),
#     )
#     figure.update_layout(margin={"r": 0.5, "t": 0.5, "l": 0, "b": 0})
#     return figure


# def get_plot_picks(message, t0, tn):
#     t0_idx = 0
#     t_picks = []
#     colors = []
#     for i, x in enumerate(message):
#         if timestamp_seconds(x["timestamp"]) >= t0:
#             if t0_idx == 0:
#                 t0_idx = i
#             if timestamp_seconds(x["timestamp"]) <= tn:
#                 t_picks.append(timestamp_seconds(x["timestamp"]) - t0)
#                 if x["type"] == "p":
#                     colors.append("b")
#                 elif x["type"] == "s":
#                     colors.append("r")
#                 else:
#                     raise("Phase type error!")
#             else:
#                 return t_picks, colors, t0_idx
#     return t_picks, colors, t0_idx

# def get_plot_events(message, t0, tn):
#     t0_idx = 0
#     t_events = []
#     mag_events = []
#     loc_events = []
#     for k, x in message.items():
#         if timestamp_seconds(x["time"]) >= t0:
#             # if t0_idx == 0:
#             #     t0_idx = i
#             if timestamp_seconds(x["time"]) <= tn - 8:
#                 t_events.append(timestamp_seconds(x["time"]) - t0)
#                 mag_events.append(x["magnitude"])
#                 loc_events.append(x["location"])
#             else:
#                 return t_events, mag_events, loc_events, t0_idx
#     return t_events, mag_events, loc_events, t0_idx


# def update_figure(figure, lat_list, lng_list, z_list, mag_events, t_events):
#     if(figure is not None):
#         figure.data = []
#     figure_df = pd.DataFrame({'lat': lat_list, 'lon': lng_list, 'z': z_list, 'mag': mag_events,
#                               'time': t_events, 'size': [(mag_event**4) / 3.5 for mag_event in mag_events]})
#     figure = px.scatter_mapbox(
#         figure_df,
#         lat="lat",
#         lon="lon",
#         hover_data=[
#             "mag",
#             "time",
#             "lat",
#             "lon"],
#         size="size",
#         color_discrete_sequence=["fuchsia"],
#         zoom=MAP_ZOOM,
#         height=300)
#     figure = update_figure_layout(figure)
#     return figure


# def update_figure_with_cols(figure, col1, col2, lat_list, lng_list, z_list, mag_events, t_events):
#     with col1:
#         figure = update_figure(figure, lat_list, lng_list, z_list, mag_events, t_events)
#     return figure


# def tweep_update_with_media(api, mag, lng, lat, z, event_time, geolocator):
#     temp_time = time.time()
#     # get figure using update_figure
#     figure = update_figure(None, [lat], [lng], [z], [mag], [event_time])
#     figure.write_image("twitter_fig.png")
#     print("Time taken to render: %f" % (time.time() - temp_time))

#     address = latlon2address(lat, lng, geolocator)

#     if address is not None:
#         caption = f"Magnitude {mag} earthquake occurred at address {address} at time {event_time}"
#     else:
#         caption = "Magnitude %f earthquake happened at longitude %f degrees, latitude %f degrees at depth %f km at time %s" % (
#             mag, lng, lat, z, event_time)

#     try:
#         api.update_with_media("twitter_fig.png", caption)
#         print('Update Twitter with media success!', flush=True)
#         global I_MADE_A_TWEET
#         I_MADE_A_TWEET = True  # Demo purpose, don't want to use up all the Twitter API Quota
#         print("Time taken to from start to end to fully upload to twitter: %f" % (time.time() - temp_time))
#     except BaseException:
#         pass


# def tweepy_status_update(event_dict):
#     if(len(event_dict) > 0):
#         event = list(event_dict.values())[-1]
#         print("tweepy_status_update (event): ", event)
#         event_time = event['time']
#         lng = lng_from_x(event['location'][0])
#         lat = lat_from_y(event['location'][1])
#         z = event['location'][2]
#         mag = event['magnitude']
#         bundle = (lng, lat, z, mag)
#         global prev_event_bundle
#         if(bundle != prev_event_bundle):
#             print("----------New Event----------")
#             prev_event_bundle = bundle

#             if mag > BOT_MAGNITUDE_THRESHOLD and api is not None and not I_MADE_A_TWEET:
#                 print("time is %s, current time is %f" % (event_time, time.time()))
#                 print("Try to update status on twitter............")
#                 print("Magnitude %f earthquake happened at longitude %f, latitude %f at depth %f at time %s" % (mag, lng, lat, z, event_time))

#                 upload_thread = threading.Thread(
#                     target=tweep_update_with_media, name="Uploader", args=(
#                         api, mag, lng, lat, z, event_time, geolocator, ))
#                 upload_thread.start()

#                 temp_time = time.time()
#                 # Pure text upload, will be fast
#                 # api.update_status(
#                 #     "Magnitude %f earthquake happened at longitude %f, latitude %f at depth %f at time %s" %
#                 #     (mag, lng, lat, z, event_time))
#                 print("Time taken for fast alert: %f" % (time.time() - temp_time))  # It took: 0.161690 seconds


# def extract_df_from_event_dict(event_dict):
#     event_dict_values = list(event_dict.values())
#     event_dict_values.reverse()
#     lat_values = []
#     lon_values = []
#     z_values = []
#     mag_values = []
#     time_values = []
#     for event in event_dict_values:
#         lon_values.append(lng_from_x(event['location'][0]))
#         lat_values.append(lat_from_y(event['location'][1]))
#         z_values.append(event['location'][2])
#         mag_values.append(event['magnitude'])
#         time_values.append(event['time'])
#     event_dict_df = pd.DataFrame({'Magnitude': mag_values, 'Time': time_values, 'Latitude (deg)': lat_values,
#                                   'Longitude (deg)': lon_values, 'Depth (km)': z_values})
#     return event_dict_df

def initialize_plot():

# # Page header
# image_data = np.asarray(Image.open('quakeflow logo design 2.jpg'))
# st.image(image_data, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
# st.balloons()

    ## Streamlit layout
    plot1, plot2, col1, col2, fig1, fig2, ax1, ax2, lines1, scatters1, scatters2 = None, None, None, None, None, None, None, None, None, None, None
    col1, col2 = st.beta_columns([1, 1])

    ## Initial plotting
    with col1:
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5.8))
        x = np.arange(WAVE_WINDOW_LENGTH // HOP_LENGTH) * (dt * HOP_LENGTH)
        ax1.set_ylim(-1, NUM_STATION)
        ax1.set_xlim(0, WAVE_WINDOW_LENGTH*dt)

        lines1 = []
        for i in STATION_IDX.values():
            line, = ax1.plot(x, np.zeros(len(x)) + i, linewidth=0.5)
            lines1.append(line)
        scatters1 = []
        for i in STATION_IDX.values():
            scatter = ax1.scatter([-1], [-1], s=300, c="white", marker="|")
            scatters1.append(scatter)
        ax1.scatter([-1], [-1], s=200, c=PICK_COLOR["p"], marker="|", label="P-wave")
        ax1.scatter([-1], [-1], s=200, c=PICK_COLOR["s"], marker="|", label="S-wave")
        ax1.legend(loc="upper left")
        ax1.title.set_text("Streaming Seismic Waveforms and Detected P/S Phases")
        plot1 = st.pyplot(fig1)


    # with col2:
    #     fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5.8))
    #     scatters2 = ax2.scatter([0], [0], s=50, c="k", marker="x", label="Earthquake")
    #     ax2.legend()
    #     ax2.set_xlim(CONFIG["xlim_degree"])
    #     ax2.set_ylim(CONFIG["ylim_degree"])
    #     ax2.title.set_text("Detected Earthquakes")
    #     plot2 = st.pyplot(fig2)

    #     experimental_df = pd.DataFrame({'lat': [], 'lon': [], 'z': [], 'mag': [], 'time': [], 'size': []})
    #     event_df = pd.DataFrame({'Magnitude': [], 'Time': [], 'Latitude (deg)': [], 'Longitude (deg)': [], 'Depth (km)': []})
    #     experimental = px.scatter_mapbox(
    #         experimental_df,
    #         lat="lat",
    #         lon="lon",
    #         hover_data=[
    #             "mag",
    #             "time",
    #             "lat",
    #             "lon"],
    #         color_discrete_sequence=["fuchsia"],
    #         zoom=MAP_ZOOM,
    #         height=300)
    #     experimental = update_figure_layout(experimental)
    #     map_figure_experimental = st.plotly_chart(experimental, width=MAP_WIDTH, height=MAP_HEIGHT)

# ui_plot = st.pyplot(plt)
# catalog_df_visual = st.empty()
# prev_time = time.time()
# prev_time_bot = time.time()    
    

    return plot1, plot2, col1, col2, fig1, fig2, ax1, ax2, lines1, scatters1, scatters2



if __name__ == "__main__":

    WAVE_WINDOW_LENGTH = 6000
    HOP_LENGTH = 2
    PICK_WINDOW_LENGTH = 6000
    EVENT_WINDOW_LENGTH = 6000
    EVENT_MIN_GAP = 5
    PLOT_CHANNEL = "HHZ"
    LATEST_TIMESTAMP = 0
    PICK_COLOR = {"p": "blue", "s": "red"}
    dt = 0.01

    STATIONS["channel"] = STATIONS["id"].apply(lambda x: x.split('.')[-1])
    STATIONS = STATIONS[STATIONS["channel"] == PLOT_CHANNEL[:-1]]
    NUM_STATION = len(STATIONS)
    STATION_IDX = dict([(c, i) for i, c in enumerate(STATIONS["id"])])

    REFRESH_SEC = 10
    prev_time = time.time()
    
    dummy = 0
    
    plot1, plot2, col1, col2, fig1, fig2, ax1, ax2, lines1, scatters1, scatters2 = initialize_plot()

    wave_vec = dict()
    wave_time = dict()
    wave_mean = dict()
    wave_std = dict()
    pick_queue = defaultdict(deque)
    pick_time = dict()
    event_queue = deque()

    # Handle messages from Kafka
    for i, message in enumerate(consumer):
        
        if message.topic == "waveform_raw":
            key = message.key.strip('"')
            if key[-len(PLOT_CHANNEL):] == PLOT_CHANNEL:
                if key not in wave_vec:
                    wave_vec[key] = deque(maxlen=WAVE_WINDOW_LENGTH)
                    wave_vec[key].extend([0 for _ in range(WAVE_WINDOW_LENGTH)])
                    wave_mean[key] = np.mean(message.value['vec'])
                    wave_std[key] = np.std(message.value['vec'])
                wave_vec[key].extend(message.value['vec'])
                timestamp = timestamp_seconds(message.value['timestamp']) + len(message.value['vec'])*dt
                wave_time[key] = timestamp
                wave_mean[key] = 0.9*wave_mean[key] + 0.1*np.mean(message.value['vec'])
                wave_std[key] = 0.5*wave_std[key] + 0.5*np.std(message.value['vec'])
                if timestamp > LATEST_TIMESTAMP:
                    LATEST_TIMESTAMP = timestamp
                # print(f"LATEST_TIMESTAMP = {LATEST_TIMESTAMP}")
                dummy += 1

        elif message.topic == "phasenet_picks":
            key = message.key
            pick = message.value
            pick_queue[key].append(pick)
            while (key in wave_time) and (timestamp_seconds(pick_queue[key][0]["timestamp"]) < wave_time[key] - PICK_WINDOW_LENGTH*dt):
                pick_queue[key].popleft()
                if len(pick_queue[key]) == 0:
                    break

        elif message.topic == "gmma_events":
            event_queue.append(message.value)
            while timestamp_seconds(event_queue[0]["time"]) < LATEST_TIMESTAMP - EVENT_WINDOW_LENGTH*dt:
                event_queue.popleft()
                if len(event_queue) == 0:
                    break

        else:
            print(message.topic)
            raise("Topic Error!")

        # if dummy > 500:
        #     print("plotting...")
        #     for key in wave_vec:
        #         y = STATION_IDX[key[:-1]]
        #         # t = np.arange(-WAVE_WINDOW_LENGTH,0)*dt 
        #         # t = wave_time[key] + np.arange(-WAVE_WINDOW_LENGTH,0)*dt 
        #         plt.plot(np.arange(len(wave_vec[key]))*dt, normalize(np.array(wave_vec[key]))/3 + STATION_IDX[key[:-1]])
        #         for pick in pick_queue[key[:-1]]:
        #             t = timestamp_seconds(pick["timestamp"]) - wave_time[key] + WAVE_WINDOW_LENGTH*dt
        #             if (t >= 0) and (t <= WAVE_WINDOW_LENGTH*dt):
        #                 plt.plot([t, t], [y-1,y+1], color=PICK_COLOR[pick["type"]])
        #     print(event_queue)
        #     plt.show()
        #     dummy = 0
            

        # # Tweepy timer
        # if time.time() - prev_time_bot > EVENT_MIN_GAP:
        #     tweepy_status_update(event_dict)
        #     prev_time_bot = time.time()

        if time.time() - prev_time > REFRESH_SEC:
            prev_time = time.time()
            print("refreshing...")

            ## update waveform plot
            for key in wave_vec:
                y = STATION_IDX[key[:-1]]
                picks, colors = [], []
                for pick in pick_queue[key[:-1]]:
                    t = timestamp_seconds(pick["timestamp"]) - wave_time[key] + WAVE_WINDOW_LENGTH*dt
                    if (t >= 0):
                        picks.append(t)
                        colors.append(PICK_COLOR[pick["type"]])
                        if (t > WAVE_WINDOW_LENGTH*dt):
                            break
                with col1:
                    lines1[y].set_ydata((np.array(wave_vec[key])[::HOP_LENGTH] - wave_mean[key])/wave_std[key]/5 + STATION_IDX[key[:-1]])
                    scatters1[y].set_offsets(np.c_[picks, np.ones_like(picks) * y])
                    scatters1[y].set_color(colors)
                    plot1.pyplot(fig1)

            ## update earthquake plot
            if len(event_queue) > 0:
                lat, lon = dict(), dict()
                for event in event_queue:
                    t = timestamp_seconds(event["time"]) - LATEST_TIMESTAMP + WAVE_WINDOW_LENGTH*dt
                    bin = np.round(t / EVENT_MIN_GAP) * EVENT_MIN_GAP ## remove duplicated events
                    if (t>=0):
                        lat[bin] = event["latitude"]
                        lon[bin] = event["longitude"]
                        if (t > WAVE_WINDOW_LENGTH*dt):
                            break
                with col2:
                    scatters2.set_offsets(np.c_[list(lon.values()), list(lat.values())])
                    plot2.pyplot(fig2)
                    
                # t_events, mag_events, loc_events, t0_idx = get_plot_events(event_dict, min_t, max_t)
                # if len(t_events) > 0:
                #     loc_events = np.array(loc_events)

                #     # organize data into the correct form
                #     lng_list, lat_list, z_list = loc_events_organize(loc_events)

                #     # update figure
                #     experimental = update_figure_with_cols(experimental, col1, col2, lat_list, lng_list, z_list, mag_events, t_events)
                #     event_df = extract_df_from_event_dict(event_dict)

        #     if len(keys) > 0:
        #         print("plotting...")
        #         with col2:
        #             ui_plot.pyplot(plt)
        #             catalog_df_visual.dataframe(event_df)
        #         with col1:
        #             map_figure_experimental.plotly_chart(experimental, width=MAP_WIDTH, height=MAP_HEIGHT)

        # if message.topic == "waveform_raw":
        #     time.sleep(0.1)
