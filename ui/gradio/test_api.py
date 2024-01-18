# %%
from gradio_client import Client
import obspy
import numpy as np
import json
import pandas as pd

# %%

waveform = obspy.read()
array = np.array([x.data for x in waveform]).T

# pipeline = PreTrainedPipeline()
inputs = array.tolist()
inputs = json.dumps(inputs)
# picks = pipeline(inputs)
# print(picks)

# %%
client = Client("ai4eps/phasenet") 
output, file = client.predict(["test_test.mseed"])
# %%
with open(output, "r") as f:
    picks = json.load(f)["data"]

# %%
picks = pd.read_csv(file)


# %%
job = client.submit(["test_test.mseed", "test_test.mseed"], api_name="/predict")  # This is not blocking

print(job.status())

# %%
output, file = job.result() 

