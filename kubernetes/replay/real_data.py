# %%
# https://docs.obspy.org/packages/autogen/obspy.clients.seedlink.easyseedlink.create_client.html#obspy.clients.seedlink.easyseedlink.create_client
from obspy.clients.seedlink.easyseedlink import create_client


# %%
def handle_data(trace):
    print("Received new data:")
    print(trace)
    print()


# %%
client = create_client("rtserve.iris.washington.edu:18000", handle_data)
client.select_stream("CI", "LRL", "HNZ")
client.run()

# %%
