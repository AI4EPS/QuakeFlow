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
    print(f"Associating on {len(request.station_id)} picks.", flush=True)
    return {
        "events": {
            "time": [],
            "latitude": [],
            "longitude": [],
            "depth_km": [],
        },
        "picks": {
            "station_id": [],
            "phase_time": [],
            "phase_type": [],
            "phase_score": [],
            "phase_amplitude": [],
            "phase_polarity": [],
        },
    }
