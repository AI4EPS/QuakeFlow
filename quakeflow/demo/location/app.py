from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Define a request body model
class Pick(BaseModel):
    station_id: list
    phase_time: list
    phase_type: list
    phase_score: list
    phase_amplitude: list
    phase_polarity: list


# Define an endpoint to make predictions
@app.post("/predict")
def predict(request: Pick):
    print(f"Locating on {len(request.station_id)} picks.", flush=True)
    return {
        "time": [],
        "latitude": [],
        "longitude": [],
        "depth_km": [],
        "num_p_picks": [],
        "num_s_picks": [],
    }
