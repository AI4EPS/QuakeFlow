# %%
import os
import fsspec
import obspy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

figure_dir = "figures"
os.makedirs(figure_dir, exist_ok=True)

# %% Two filesystems: GCS for picks, S3 for waveforms
fs_gcs = fsspec.filesystem("gs", token="anon")
bucket = "quakeflow_catalog"

fs_s3 = fsspec.filesystem("s3", anon=True)

# %% Browse available data
print("Regions:", [x.split("/")[-1] for x in fs_gcs.ls(bucket)])

years = fs_gcs.ls(f"{bucket}/SC/phasenet_merged")
print(f"\nSC phasenet_merged years: {[y.split('/')[-1] for y in sorted(years)]}")

# %% Read picks — Ridgecrest M6.4 foreshock, 2019-07-04 (jday 185)
region = "SC"
year = 2019
jday = 185  # July 4, 2019

picks_path = f"{bucket}/{region}/phasenet_merged/{year}/{year}.{jday:03d}.csv"
print(f"Reading picks from: gs://{picks_path}")
with fs_gcs.open(picks_path, "r") as f:
    picks = pd.read_csv(f, dtype=str)

picks["phase_score"] = picks["phase_score"].astype(float)
picks["phase_time"] = pd.to_datetime(picks["phase_time"])

print(f"Total picks: {len(picks)}, Unique stations: {picks['station_id'].nunique()}")
print(f"Columns: {list(picks.columns)}")

# %% Filter picks for station CI.CLC
station_id = "CI.CLC..BH"
station_picks = picks[picks["station_id"] == station_id].copy()
print(f"\nPicks at {station_id}: {len(station_picks)}")


# %% Read waveforms from SCEDC / NCEDC S3
# SCEDC: s3://scedc-pds/continuous_waveforms/{year}/{year}_{jday}/{net}{sta:5s}{cha}_{loc}_{year}{jday}.ms
# NCEDC: s3://ncedc-pds/continuous_waveforms/{net}/{year}/{year}.{jday}/{sta}.{net}.{cha}.{loc}.D.{year}.{jday}
def read_waveform_scedc(network, station, channel, location, year, jday, fs=fs_s3):
    loc_str = location if location and location not in ("", "--") else "_"
    sta_padded = station.ljust(5, "_")
    path = (
        f"scedc-pds/continuous_waveforms/"
        f"{year}/{year}_{jday:03d}/"
        f"{network}{sta_padded}{channel}_{loc_str}_{year}{jday:03d}.ms"
    )
    print(f"  s3://{path}")
    with fs.open(path, "rb") as f:
        return obspy.read(f)


def read_waveform_ncedc(network, station, channel, location, year, jday, fs=fs_s3):
    loc_str = location if location and location not in ("", "--") else ""
    path = (
        f"ncedc-pds/continuous_waveforms/"
        f"{network}/{year}/{year}.{jday:03d}/"
        f"{station}.{network}.{channel}.{loc_str}.D.{year}.{jday:03d}"
    )
    print(f"  s3://{path}")
    with fs.open(path, "rb") as f:
        return obspy.read(f)


# %%
parts = station_id.split(".")
network, station, location, instrument = parts[0], parts[1], parts[2], parts[3]

print(f"Reading waveforms for {network}.{station}:")
st = obspy.Stream()
for comp in ["E", "N", "Z"]:
    channel = f"{instrument}{comp}"
    try:
        if region == "SC":
            st += read_waveform_scedc(network, station, channel, location, year, jday)
        else:
            st += read_waveform_ncedc(network, station, channel, location, year, jday)
    except Exception as e:
        print(f"  Could not read {channel}: {e}")

st.merge(fill_value="latest")
print(f"\n{st}")

# %% Find pick closest to M6.4 origin time, then plot +/-30s window
ridgecrest_ot = pd.Timestamp("2019-07-04 17:33:49")
station_picks["_dt"] = (station_picks["phase_time"] - ridgecrest_ot).abs()
pick = station_picks.sort_values("_dt").iloc[0]

pick_time = pick["phase_time"]
print(f"Selected pick: {pick['phase_type']}  time={pick_time}  score={pick['phase_score']:.3f}")

window = pd.Timedelta(seconds=30)
nearby_picks = station_picks[
    (station_picks["phase_time"] >= pick_time - window)
    & (station_picks["phase_time"] <= pick_time + window)
]
print(f"\nPicks within +/-30s:")
print(nearby_picks[["station_id", "phase_type", "phase_time", "phase_score"]].to_string(index=False))

# %% Plot
pick_utc = obspy.UTCDateTime(pick_time)
st_plot = st.copy().trim(pick_utc - 30, pick_utc + 30)

fig, axes = plt.subplots(len(st_plot), 1, figsize=(14, 2.5 * len(st_plot)), sharex=True)
if len(st_plot) == 1:
    axes = [axes]

colors = {"P": "r", "S": "b"}
for ax, tr in zip(axes, st_plot):
    times = tr.times("matplotlib")
    ax.plot(times, tr.data, "k-", linewidth=0.5)
    ax.set_ylabel(tr.stats.channel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

for _, p in nearby_picks.iterrows():
    p_utc = obspy.UTCDateTime(p["phase_time"])
    color = colors.get(p["phase_type"], "g")
    label = f"{p['phase_type']} ({p['phase_score']:.2f})"
    for ax in axes:
        ax.axvline(p_utc.matplotlib_date, color=color, linewidth=1.2, alpha=0.8, label=label)

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[0].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time (UTC)")
fig.suptitle(f"{station_id} | Ridgecrest M6.4 | {pick_time.strftime('%Y-%m-%d')} | {len(nearby_picks)} picks in +/-30s")
plt.tight_layout()
fig_path = f"{figure_dir}/waveform_{station_id}_{year}_{jday:03d}.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved: {fig_path}")
plt.close()

# %%
