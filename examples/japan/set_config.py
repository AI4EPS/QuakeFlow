# %%
import json

config = {
    "minlatitude": 36.8,
    "maxlatitude": 38.2,
    "minlongitude": 136.2,
    "maxlongitude": 138.3,
    "starttime": "2024-01-01T00:00:00",
    "endtime": "2024-02-29T23:00:00",
}

# %%
with open("local/hinet/config.json", "w") as f:
    json.dump(config, f, indent=2)

# %%
