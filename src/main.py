from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
import joblib
import os
import pandas as pd

from preprocess_data import extract_features


app = FastAPI()

class InputData(BaseModel):
    id: int
    mission: str

@app.on_event("startup")
def load_model():
    global model
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "your-container"
    blob_name = "your_model.pkl"
    local_model_path = "model.pkl"

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open(local_model_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    model = joblib.load(local_model_path)
    
class InputData(BaseModel):
    id: int
    mission: str = "Kepler"
    planet_star_radius_ratio: float | None = None
    a_by_rstar: float | None = None
    inclination_deg: float | None = None

@app.post("/predict")
def predict(data: InputData):
    features = extract_features(
        kicid=data.id,
        mission=data.mission,
        planet_star_radius_ratio=data.planet_star_radius_ratio,
        a_by_rstar=data.a_by_rstar,
        inclination_deg=data.inclination_deg
    )

    if all(pd.isna(value) for value in features.values()):
        raise HTTPException(status_code=404, detail="Could not extract features for the given ID.")

    df_input = pd.DataFrame([features])

    try:
        df_input["prediction"] = model.predict(df_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    result = {
        "id": data.id,
        "mission": data.mission,
        "prediction": df_input["prediction"].iloc[0],
        "features": {k: (None if pd.isna(v) else v) for k, v in features.items()}
    }
    return result