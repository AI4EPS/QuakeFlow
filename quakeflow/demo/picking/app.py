from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Define a request body model
class Data(BaseModel):
    id: list
    vec: list
    timestamp: list


# Define an endpoint to make predictions
@app.post("/predict")
def predict(request: Data):
    print(f"Picking on {len(request.id)} stations.", flush=True)
    return {
        "station_id": [],
        "phase_time": [],
        "phase_type": [],
        "phase_score": [],
        "phase_amplitude": [],
        "phase_polarity": [],
    }
