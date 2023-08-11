# %%
import fsspec
import pandas as pd
import obspy
import matplotlib.pyplot as plt

# %%
protocal = "gs"
bucket = "quakeflow_share"
folder = "demo/obspy"

# %% Seismic stations used in the demo
stations = pd.read_csv(f"{protocal}://{bucket}/{folder}/stations.csv")
plt.figure()
plt.scatter(stations["longitude"], stations["latitude"], marker="^", label="stations")
plt.show()

# %% Read replay waveforms
fs = fsspec.filesystem(protocal)
mseeds = fs.glob(f"{bucket}/{folder}/waveforms/*/*.mseed")

# %%
for mseed in mseeds:
    print(mseed)
    with fs.open(mseed, "rb") as f:
        st = obspy.read(f)

        # plot a few seconds
        tmp = st.slice(starttime=st[0].stats.starttime, endtime=st[0].stats.starttime + 20)
        tmp.plot()
    break
