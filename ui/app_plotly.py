import datetime as dt
import os
import pathlib
import threading
from collections import defaultdict, deque
from datetime import datetime
from json import loads

import dash
import numpy as np
import pandas as pd
import plotly.express as px

# import dash_core_components as dcc
# import dash_html_components as html
# import dash_table
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from kafka import KafkaConsumer

##############################  Kakfa #################################################

consumer = None
# Connection to Kafka
try:
    print('Connecting to k8s kafka')
    BROKER_URL = 'quakeflow-kafka-headless:9092'
    # BROKER_URL = "34.83.137.139:9094"
    consumer = KafkaConsumer(
        bootstrap_servers=[BROKER_URL],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        key_deserializer=lambda x: loads(x.decode('utf-8')),
        value_deserializer=lambda x: loads(x.decode('utf-8')),
    )
    consumer.subscribe(['phasenet_picks', 'gmma_events', 'waveform_raw', 'waveform_phasenet'])
    use_kafka = True
    print('k8s kafka connection success!')
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
            value_deserializer=lambda x: loads(x.decode('utf-8')),
        )
        consumer.subscribe(['phasenet_picks', 'gmma_events', 'waveform_raw', 'waveform_phasenet'])
        use_kafka = True
        print('local kafka connection success!')
    except BaseException:
        print('local Kafka connection error')

if not consumer:
    raise ('No kafka server found!')

timestamp_seconds = lambda x: datetime.fromisoformat(x).timestamp()

wave_vec = dict()
wave_time = dict()
wave_mean = dict()
wave_std = dict()
pick_queue = defaultdict(deque)
pick_time = dict()
event_queue = deque()

PLOT_CHANNEL = "HHZ"
WAVE_WINDOW_LENGTH = 3000
PICK_WINDOW_LENGTH = WAVE_WINDOW_LENGTH * 2 * 60 * 24 * 30
EVENT_WINDOW_LENGTH = WAVE_WINDOW_LENGTH * 2 * 60 * 24 * 30
DELTAT = 0.01
PICK_COLOR = {"p": "blue", "s": "red"}


def run_consumser():
    global wave_vec
    global wave_time
    global wave_mean
    global wave_std
    global pick_queue
    global event_queue
    latest_timestamp = 0
    # timestamp_seconds = lambda x: datetime.fromisoformat(x).timestamp()
    for _, message in enumerate(consumer):

        if message.topic == "waveform_raw":
            continue
            key = message.key.strip('"')
            key = key
            if key[-len(PLOT_CHANNEL) :] == PLOT_CHANNEL:
                if key not in wave_vec:
                    wave_vec[key] = deque(maxlen=WAVE_WINDOW_LENGTH)
                    wave_vec[key].extend([0 for _ in range(WAVE_WINDOW_LENGTH)])
                    wave_mean[key] = np.mean(message.value['vec'])
                    wave_std[key] = np.std(message.value['vec'])
                wave_vec[key].extend(message.value['vec'])
                timestamp = timestamp_seconds(message.value['timestamp']) + len(message.value['vec']) * DELTAT
                wave_time[key] = timestamp
                # wave_mean[key] = 0.5 * wave_mean[key] + 0.5 * np.mean(message.value['vec'])
                # wave_std[key] = 0.5 * wave_std[key] + 0.5 * np.std(message.value['vec'])
                wave_mean[key] = np.mean(message.value['vec'])
                wave_std[key] = np.std(message.value['vec'])
                if timestamp > latest_timestamp:
                    latest_timestamp = timestamp

        elif message.topic == "waveform_phasenet":
            key = message.key.strip('"')
            key = key + PLOT_CHANNEL[-1]
            if key[-len(PLOT_CHANNEL) :] == PLOT_CHANNEL:
                if key not in wave_vec:
                    wave_vec[key] = deque(maxlen=WAVE_WINDOW_LENGTH)
                    wave_vec[key].extend([0 for _ in range(WAVE_WINDOW_LENGTH)])
                    wave_mean[key] = np.mean(message.value['vec'])
                    wave_std[key] = np.std(message.value['vec'])
                wave_vec[key].extend(np.array(message.value['vec'])[..., -1].tolist())
                timestamp = timestamp_seconds(message.value['timestamp']) + len(message.value['vec']) * DELTAT
                wave_time[key] = timestamp
                wave_mean[key] = 0.5 * wave_mean[key] + 0.5 * np.mean(message.value['vec'])
                wave_std[key] = 0.5 * wave_std[key] + 0.5 * np.std(message.value['vec'])
                if timestamp > latest_timestamp:
                    latest_timestamp = timestamp

        elif message.topic == "phasenet_picks":
            key = message.key
            pick = message.value
            pick_queue[key].append(pick)
            while (key in wave_time) and (
                timestamp_seconds(pick_queue[key][0]["timestamp"]) < wave_time[key] - PICK_WINDOW_LENGTH * DELTAT
            ):
                pick_queue[key].popleft()
                if len(pick_queue[key]) == 0:
                    break

        elif message.topic == "gmma_events":
            event_queue.append(message.value)
            while timestamp_seconds(event_queue[0]["time"]) < latest_timestamp - EVENT_WINDOW_LENGTH * DELTAT:
                event_queue.popleft()
                if len(event_queue) == 0:
                    break

        else:
            print(message.topic)
            raise ("Topic Error!")


# lock = threading.Lock()
p = threading.Thread(target=run_consumser, args=())
p.start()


##############################  Ploty #################################################

GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "QuakeFlow Dashboard"

server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

app.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("QuakeFlow Earthquake Monitoring", className="app__header__title"),
                        html.P(
                            # "Realtime earthquake waveforms and detections at Hawaii",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("SOURCE CODE", className="link-button"),
                            href="https://wayneweiqiang.github.io/QuakeFlow/",
                        ),
                        # html.A(
                        #     html.Button("ENTERPRISE DEMO", className="link-button"),
                        #     href="https://plotly.com/get-demo/",
                        # ),
                        # html.A(
                        #     html.Img(
                        #         src=app.get_asset_url("dash-new-logo.png"),
                        #         className="app__menu__img",
                        #     ),
                        #     href="https://plotly.com/dash/",
                        # ),
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                # seismic waveform
                html.Div(
                    [
                        html.Div([html.H6("Realtime Seismic Waveform", className="graph__title")]),
                        dcc.Graph(
                            id="seismic-waveform",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),
                        dcc.Interval(
                            id="seismic-waveform-update",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                    ],
                    className="two-thirds column wind__speed__container",
                ),
                html.Div(
                    [
                        # earthquake map
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Earthquake Map",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="earthquake-map",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container first",
                        ),
                        # earthquake table
                        html.Div(
                            [
                                html.Div([html.H6("Earthquake Information", className="graph__title")]),
                                dash_table.DataTable(
                                    id="earthquake-table",
                                    data=[],
                                    page_action='none',
                                    style_table={'height': '300px', 'overflowY': 'auto'},
                                    style_header={'backgroundColor': 'rgb(50, 50, 50)'},
                                    style_cell={'backgroundColor': app_color["graph_bg"], 'color': 'white'},
                                ),
                            ],
                            className="graph__container second",
                        ),
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


@app.callback(Output("seismic-waveform", "figure"), [Input("seismic-waveform-update", "n_intervals")])
def gen_waveform(interval):
    """
    Generate seismic waveforms.

    :params interval: update the graph based on an interval
    """

    traces = []
    keys = sorted(list(wave_vec.keys()))
    num = len(keys)

    for i, k in enumerate(keys):
        # waveform
        trace = dict(
            type="scatter",
            y=list((np.array(wave_vec[k]) - wave_mean[k]) / wave_std[k] / 4 + num - i),
            line={
                "color": "C{i}",
                "width": 1.0,
            },
            hoverinfo="skip",
            mode="lines",
            name=k,
        )
        traces.append(trace)

        # picks
        picks = pd.DataFrame(list(pick_queue[k[:-1]]))
        if len(picks) > 0:
            picks["t"] = picks["timestamp"].apply(
                lambda x: (timestamp_seconds(x) - wave_time[k]) / DELTAT + WAVE_WINDOW_LENGTH
            )
            x = picks[picks["type"] == "p"]["t"]
            trace = dict(
                type="scatter",
                x=x,
                y=[num - i for _ in range(len(x))],
                hoverinfo="skip",
                mode="markers",
                marker={
                    "color": "rgb(0,255,0)",
                    'line': {'color': 'rgb(0,255,0)', 'width': 2},
                    "size": 20,
                    "symbol": [42 for _ in range(len(x))],
                },
                name=None,
                showlegend=False,
            )
            traces.append(trace)
            x = picks[picks["type"] == "s"]["t"]
            trace = dict(
                type="scatter",
                x=x,
                y=[num - i for _ in range(len(x))],
                hoverinfo="skip",
                mode="markers",
                marker={
                    "color": "rgb(255,0,0)",
                    'line': {'color': 'rgb(255,0,0)', 'width': 2},
                    "size": 20,
                    "symbol": [42 for _ in range(len(x))],
                },
                name=None,
                showlegend=False,
            )
            traces.append(trace)

    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        height=700,
        xaxis={
            # "range": [0, 6000],
            # "tickvals": [0, 1000, 2000, 3000, 4000, 5000, 6000],
            # "ticktext": ["60", "50", "40", "30", "20", "10", "0"],
            "range": [0, 3000],
            "tickvals": [0, 1000, 2000, 3000],
            "ticktext": ["30", "20", "10", "0"],
            "showline": True,
            "zeroline": False,
            "fixedrange": True,
            "title": "Time Elapsed (sec)",
        },
        yaxis={
            "range": [
                -1.0,
                num + 1.0,
            ],
            "showgrid": False,
            "showline": True,
            "fixedrange": True,
            "zeroline": False,
            "gridcolor": app_color["graph_line"],
            "nticks": num,
        },
    )

    return dict(data=traces, layout=layout)


def remove_duplicates(df):
    MIN_DT = 3.0
    df["key"] = df['time'].apply(lambda x: round(datetime.fromisoformat(x).timestamp() / MIN_DT))
    df = df.groupby(df['key']).aggregate('last')
    return df


@app.callback(Output("earthquake-map", "figure"), [Input("seismic-waveform-update", "n_intervals")])
def gen_wind_direction(interval):
    """
    Generate earthquake map.

    :params interval: update the graph based on an interval
    """

    events = pd.DataFrame(list(event_queue))
    if len(events) > 0:
        events = remove_duplicates(events)
        events.sort_values(by=['time'], ascending=False, inplace=True)
        events["size"] = events["magnitude"].apply(lambda x: (x + 2) / 10)
        fig = px.scatter_mapbox(
            events,
            lat="latitude",
            lon="longitude",
            hover_data=["time", "magnitude"],
            color_discrete_sequence=["fuchsia"],
            # size="size",
            zoom=6,
            height=300,
            opacity=0.5,
        )
    else:
        tmp = pd.DataFrame({"time": [None], "latitude": [0], "longitude": [0], "magnitude": [None]})
        fig = px.scatter_mapbox(
            tmp, lat="latitude", lon="longitude", color_discrete_sequence=["black"], zoom=0, height=300
        )

    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": 'traces',
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ],
            }
        ],
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


@app.callback(
    [Output("earthquake-table", "data"), Output('earthquake-table', 'columns')],
    [Input("seismic-waveform-update", "n_intervals")],
)
def gen_wind_direction(interval):
    """
    Generate earthquake table.

    :params interval: update the graph based on an interval
    """

    columns = ["time", "magnitude", "latitude", "longitude"]
    events = pd.DataFrame(list(event_queue))
    if len(events) > 0:
        events["magnitude"] = events["magnitude"].apply(lambda x: round(x, 1))
        events["latitude"] = events["latitude"].apply(lambda x: round(x, 2))
        events["longitude"] = events["longitude"].apply(lambda x: round(x, 2))
        events = remove_duplicates(events)
        events.sort_values(by=['time'], ascending=False, inplace=True)
        data = events[columns].to_dict("records")
        return data, [{"name": i, "id": i} for i in columns]
    else:
        return [], [{"name": i, "id": i} for i in columns]


if __name__ == "__main__":
    # p = threading.Thread(target=run_consumser, args=())
    # p.start()
    app.run_server(debug=True)
