# %%
config = {
    "degree2km": 111.1949,
    "provider": None,
    "network": None,
    "station": None,
    "channel": "HH*,BH*,EH*,HN*",
    "channel_priorities": (
        "HH[ZNE12]",
        "BH[ZNE12]",
        "MH[ZNE12]",
        "EH[ZNE12]",
        "LH[ZNE12]",
        "HL[ZNE12]",
        "BL[ZNE12]",
        "ML[ZNE12]",
        "EL[ZNE12]",
        "LL[ZNE12]",
        "SH[ZNE12]",
    ),
    "location_priorities": (
        "",
        "00",
        "10",
        "01",
        "20",
        "02",
        "30",
        "03",
        "40",
        "04",
        "50",
        "05",
        "60",
        "06",
        "70",
        "07",
        "80",
        "08",
        "90",
        "09",
    ),
    "level": "response",
    "phasenet": {},
    "gamma": {},
    "cctorch": {},
    "adloc": {},
    "hypodd": {},
    "growclust": {},
}

# %%
config_region = {}

region = "Ridgecrest"
config_region[region] = {
    "region": region,
    "center": [-117.504, 35.705],
    "center": [-117.504, 35.705],
    "longitude0": -117.504,
    "latitude0": 35.705,
    "xlim_degree": [-118.504, -116.504],
    "ylim_degree": [34.705, 36.705],
    "maxradius_degree": 1.0,
    "horizontal_degree": 1.0,
    "vertical_degree": 1.0,
    # "starttime": "2019-07-04T17:00:00",
    # "endtime": "2019-07-04T19:00:00",
    "starttime": "2019-07-04T00:00:00",
    "endtime": "2019-07-10T00:00:00",
    "channel_priorities": [
        "HH[321ENZ]",
        "EH[321ENZ]",
        "HN[321ENZ]",
        "BH[321ENZ]",
    ],
    "provider": ["SCEDC", "IRIS"],
}

region = "Hawaii_Loa"
config_region[region] = {
    "region": region,
    "center": [
        -155.602737,
        -155.602737,
        19.451827,
    ],
    "longitude0": -155.602737,
    "latitude0": 19.451827,
    "minlatitude": 19.451827 - 0.3 / 2,
    "maxlatitude": 19.451827 + 0.3 / 2,
    "minlongitude": -155.602737 - 0.3 / 2,
    "maxlongitude": -155.602737 + 0.3 / 2,
    "xlim_degree": [-155.602737 - 0.3 / 2, -155.602737 + 0.3 / 2],
    "ylim_degree": [19.451827 - 0.3 / 2, 19.451827 + 0.3 / 2],
    "minlatitude": 19.451827 - 0.3 / 2,
    "maxlatitude": 19.451827 + 0.3 / 2,
    "minlongitude": -155.602737 - 0.3 / 2,
    "maxlongitude": -155.602737 + 0.3 / 2,
    "xlim_degree": [-155.602737 - 0.3 / 2, -155.602737 + 0.3 / 2],
    "ylim_degree": [19.451827 - 0.3 / 2, 19.451827 + 0.3 / 2],
    "maxradius_degree": 0.3,
    "horizontal_degree": 0.3,
    "vertical_degree": 0.3,
    # "starttime": "2019-07-04T17:00:00",
    # "endtime": "2019-07-04T19:00:00",
    "starttime": "2022-11-30T22:00:00",
    "endtime": "2022-11-30T23:00:00",
    # "endtime": "2023-04-27T00:00:00",
    "channel_priorities": [
        "HH[321ENZ]",
        "BH[321ENZ]",
        "EH[321ENZ]",
        "HN[321ENZ]",
    ],
    "provider": ["IRIS"],
}

region = "South_Pole2"
config_region[region] = {
    "region": region,
    "starttime": "2003-01-01T00:00:00",
    # "starttime": "2020-05-01T00:00:00",
    # "endtime": "2020-06-01T00:00:00",
    # "starttime": "2023-04-01T00:00:00",
    "endtime": "2023-05-01T00:00:00",
    # "minlatitude": -90,
    # "maxlatitude": -80,
    # "minlongitude": -180,
    # "maxlongitude": 180,
    "network": "IU",
    "station": "QSPA",
    "provider": ["IRIS"],
    "channel_priorities": [
        "HH[321ENZ]",
        "BH[321ENZ]",
        # "EH[321ENZ]",
        # "HN[321ENZ]",
    ],
    "degree2km": 111.19492474777779,
}

region = "Montezuma"
config_region[region] = {
    "region": region,
    "starttime": "2020-05-01T00:00:00",
    "endtime": "2023-05-01T00:00:00",
    "minlatitude": 37.9937,
    "maxlatitude": 38.1657,
    "minlongitude": -122.0325,
    "maxlongitude": -121.7275,
    "provider": None,
    "network": None,
    "station": None,
    "degree2km": 111.19492474777779,
}

region = "Kilauea"
config_region[region] = {
    "region": region,
    "starttime": "2018-04-29T00:00:00",
    "endtime": "2018-08-12T00:00:00",
    "minlatitude": 19.41 - 0.1,
    "maxlatitude": 19.41 + 0.1,
    "minlongitude": -155.28 - 0.1,
    "maxlongitude": -155.28 + 0.1,
    "provider": ["IRIS"],
    "degree2km": 111.19492474777779,
    "gamma": {"zmin_km": -1, "zmax_km": 10},
}

region = "Kilauea_debug"
config_region[region] = {
    "region": region,
    "starttime": "2018-04-29T00:00:00",
    "endtime": "2018-04-29T01:00:00",
    "minlatitude": 19.41 - 0.1,
    "maxlatitude": 19.41 + 0.1,
    "minlongitude": -155.28 - 0.1,
    "maxlongitude": -155.28 + 0.1,
    "provider": ["IRIS"],
    "degree2km": 111.19492474777779,
}

region = "Kilauea_debug_v2"
config_region[region] = {
    "region": region,
    "starttime": "2018-04-29T00:00:00",
    "endtime": "2018-04-29T01:00:00",
    "minlatitude": 19.41 - 0.1,
    "maxlatitude": 19.41 + 0.1,
    "minlongitude": -155.28 - 0.1,
    "maxlongitude": -155.28 + 0.1,
    "provider": ["IRIS"],
    "degree2km": 111.19492474777779,
}

region = "BayArea"
config_region[region] = {
    "region": region,
    # "starttime": "2022-01-01T00:00:00",
    "starttime": "2022-11-01T00:00:00",
    "endtime": "2023-01-01T00:00:00",
    "minlatitude": 36.5,
    "maxlatitude": 37.0,
    "minlongitude": -121.7,
    "maxlongitude": -121.2,
    "provider": ["NCEDC"],
    "network": "BK,NC",
    "degree2km": 111.19492474777779,
    "channel": "HN*",
    "level": "channel",
    "channel_priorities": (
        "HH[ZNE12]",
        "BH[ZNE12]",
    ),
}
