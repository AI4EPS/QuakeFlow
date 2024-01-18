# %%
import os
from datetime import datetime
from io import StringIO

# %%
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# %%
url = "https://ncedc.org/cgi-bin/catalog-search2.pl"

content = {
    "format": "ncfpraw",
    "mintime": "1966/01/01,00:00:00",
    "maxtime": "2023/01/01,00:00:00",
    "minmag": None,
    "maxmag": None,
    "mindepth": None,
    "maxdepth": None,
    # "minlat": 34.5,
    # "maxlat": 42,
    # "minlon": -126.0,
    # "maxlon": -117.76,
    "minlat": None,
    "maxlat": None,
    "minlon": None,
    "maxlon": None,
    "etype": "E",
    "keywds": None,
    "outputloc": "web",
    "searchlimit": None,
}


# %%
# Define column specifications based on your input
colspecs = [
    (0, 4),
    (4, 6),
    (6, 8),
    (9, 11),
    (11, 13),
    (13, 19),
    (19, 22),
    (22, 23),
    (23, 28),
    (28, 32),
    (32, 33),
    (33, 38),
    (38, 45),
    (47, 52),
    (52, 55),
    (55, 59),
    (59, 64),
    (64, 69),
    (69, 74),
    (74, 79),
    (81, 82),
    (83, 86),
    (87, 89),
    (89, 93),
    (95, 99),
    (100, 103),
    (104, 109),
    (110, 114),
    (115, 119),
    (121, 123),
    (124, 126),
    (127, 129),
    (129, 130),
    (130, 131),
    (131, 141),
]

# Define column names https://ncedc.org/pub/doc/cat5/ncsn.mech.txt
colnames = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Seconds",
    "Latitude (deg)",
    "S or N",
    "Latitude (min)",
    "Longitude (deg)",
    "E or W",
    "Longitude (min)",
    "Depth (km)",
    "Magnitude",
    "Number of P & S times",
    "Maximum Azimuthal Gap",
    "Distance to Nearest Station",
    "RMS Travel Time Residual",
    "Horizontal Error",
    "Vertical Error",
    "Hypoinverse Code",
    "Dip Direction",
    "Dip Angle",
    "Rake",
    "Solution Misfit",
    "Number of First Motion Observations",
    "Solution Misfit + 90% Confidence",
    "Station Distribution Ratio",
    "Machine/Hand Picks",
    "Max Half-Width of 90% Confidence Range of Strike",
    "Max Half-Width of 90% Confidence Range of Dip",
    "Max Half-Width of 90% Confidence Range of Rake",
    "Convergence Flag",
    "Multiple Solution Flag",
    "Event ID",
]

colnames = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Seconds",
    "Latitude (deg)",
    "South",
    "Latitude (min)",
    "Longitude (deg)",
    "East",
    "Longitude (min)",
    "depth_km",
    "magnitude",
    "num_p/s",
    "azimuthal_gap",
    "nearest_station_km",
    "rms_travel_time",
    "x_error",
    "z_error",
    "hypoinverse_code",
    "dip_direction",
    "dip",
    "rake",
    "misfit",
    "num_first_motion",
    "90%_confidence",
    "station_ratio",
    "auto/manual_picks",
    "90%_strike",
    "90%_dip",
    "90%_rake",
    "convergence_flag",
    "multiple_solution",
    "event_id",
]

if __name__ == "__main__":
    result_path = "dataset/catalog_fm"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for year in tqdm(range(1966, 2023)):
        content["mintime"] = f"{year}/01/01,00:00:00"
        content["maxtime"] = f"{year + 1}/01/01,00:00:00"

        x = requests.post(url, data=content)

        # %%
        data = BeautifulSoup(x.text)

        # with open(os.path.join(result_path, f"{year}.txt"), "w") as f:
        #     f.write(data.text)

        # check data.text end with "No matches to your search criteria"
        if data.text.rstrip().endswith("No matches to your search criteria"):
            continue

        df = pd.read_fwf(StringIO(data.text), colspecs=colspecs, header=None, names=colnames, skiprows=7, dtype=str)

        # # Read the file
        # df = pd.read_fwf("mechanism.txt", colspecs=colspecs, header=None, names=colnames, skiprows=7, dtype=str)

        # %%
        df["datetime"] = df.apply(
            lambda x: datetime.strptime(
                f"{x['Year']}-{x['Month']}-{x['Day']}T{x['Hour']}:{x['Minute']}:{x['Seconds']}", "%Y-%m-%dT%H:%M:%S.%f"
            ),
            axis=1,
        )
        df["Longitude"] = df.apply(
            lambda x: -round(float(x["Longitude (deg)"]) + float(x["Longitude (min)"]) / 60.0, 5), axis=1
        )
        df["Latitude"] = df.apply(
            lambda x: round(float(x["Latitude (deg)"]) + float(x["Latitude (min)"]) / 60.0, 5), axis=1
        )
        df["Latitude"] = df.apply(lambda x: -1 * x["Latitude"] if x["South"] == "S" else x["Latitude"], axis=1)
        df["Longitude"] = df.apply(lambda x: -1 * x["Longitude"] if x["East"] == "E" else x["Longitude"], axis=1)
        df["strike"] = df.apply(lambda x: (float(x["dip_direction"]) - 90) % 360, axis=1)
        df.drop(
            columns=[
                "Year",
                "Month",
                "Day",
                "Hour",
                "Minute",
                "Seconds",
                "Latitude (deg)",
                "Latitude (min)",
                "Longitude (deg)",
                "Longitude (min)",
                "South",
                "East",
                "dip_direction",
            ],
            inplace=True,
        )
        df["event_id"] = df.apply(lambda x: f"nc{x['event_id']}", axis=1)

        df.to_csv(os.path.join(result_path, f"{year}.csv"), index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
