from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from typing import Dict
from src.train_model.train_model import train_model
from fastapi.responses import JSONResponse
import joblib
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os
import pandas as pd
import numpy as np

from src.preprocess_data import extract_features, create_feature_dataset_in_batches

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Aux
def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj

# Data Models
class TrainRequest(BaseModel):
    model_type: str
    params: Dict

class DataInput(BaseModel):
    id: str
    model_type: str

class PredictRequest(BaseModel):
    kicid: str
    model_type: str
    mission: str
    planet_star_radius_ratio: Optional[float] = None
    a_by_rstar: Optional[float] = None
    inclination_deg: Optional[float] = None


class PredictBatchRequest(BaseModel):
    csv_blob_path: str
    model_type: str

# Start
models = {}

@app.on_event("startup")
def load_models():
    global models

    account_url = os.getenv("AZURE_ACCOUNT_URL")
    sas_token = os.getenv("AZURE_SAS_TOKEN")
    container_name = "artifacts"

    if not account_url or not sas_token:
        raise RuntimeError("Missing AZURE_ACCOUNT_URL or AZURE_SAS_TOKEN.")

    # Connect to the blob service
    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs under the "models/" folder
    blob_list = container_client.list_blobs(name_starts_with="models/")

    for blob in blob_list:
        blob_name = blob.name  # e.g., "models/xgb_model.joblib"
        model_type = blob_name.split("/")[-1].split("_")[0]  # "xgb", "lgbm", etc.

        print(f"Loading model: {model_type} from blob: {blob_name}")

        # Download blob to memory
        blob_client = container_client.get_blob_client(blob=blob_name)
        blob_data = blob_client.download_blob().readall()

        # Load model from bytes
        model = joblib.load(BytesIO(blob_data))
        models[model_type] = model

    print("All models loaded:", list(models.keys()))

# Endpoints
@app.get("/")
def read_root():
    return {"Health": "Ok"}

@app.post("/trainer")
def trainer(request: TrainRequest):
    model_type = request.model_type.lower()
    params = request.params

    try:
        train_model(model_type, params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(
            status_code=200,
            content={"message": f"{model_type.upper()} model trained and saved successfully."}
        )

@app.post("/predict-realtime")
def predict_realtime(input: PredictRequest):
    model_type = input.model_type.lower()
    
    # Special handling for ensemble
    is_ensemble = model_type == "ensemble"
    
    if not is_ensemble and model_type not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' is not loaded.")

    # Extract features
    features = extract_features(
        kicid=input.kicid,
        mission=input.mission,
        planet_star_radius_ratio=input.planet_star_radius_ratio,
        a_by_rstar=input.a_by_rstar,
        inclination_deg=input.inclination_deg
    )

    feature_df = pd.DataFrame([features])

    try:
        if is_ensemble:
            required_models = ["xgb", "lgbm", "catboost", "nn"]
            missing_models = [m for m in required_models if m not in models]
            if missing_models:
                raise HTTPException(status_code=500, detail=f"Missing models for ensemble: {missing_models}")
            
            probs = []
            for m in required_models:
                model = models[m]
                proba = (
                    model.predict_proba(feature_df)[0][1]
                    if hasattr(model, "predict_proba")
                    else model.predict(feature_df)[0]  # fallback
                )
                probs.append(proba)

            # Average probabilities
            prediction_proba = sum(probs) / len(probs)
            prediction = int(prediction_proba >= 0.5)

        else:
            model = models[model_type]
            prediction = model.predict(feature_df)[0]
            prediction_proba = (
                model.predict_proba(feature_df)[0][1]
                if hasattr(model, "predict_proba")
                else None
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "features": convert_numpy_types(features),
        "prediction": int(prediction),
        "prediction_proba": float(prediction_proba) if prediction_proba is not None else None
    }



@app.post("/predict-batch")
def predict_batch(input: PredictBatchRequest):
    model_type = input.model_type.lower()
    is_ensemble = model_type == "ensemble"

    if not is_ensemble and model_type not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_type}' is not loaded.")

    account_url = os.getenv("AZURE_ACCOUNT_URL")
    sas_token = os.getenv("AZURE_SAS_TOKEN")
    container_name = "artifacts"

    if not account_url or not sas_token:
        raise HTTPException(status_code=500, detail="Missing Azure credentials")

    # Load CSV from Azure Blob Storage
    blob_service = BlobServiceClient(account_url=account_url, credential=sas_token)
    blob_client = blob_service.get_blob_client(container=container_name, blob=input.csv_blob_path)

    try:
        blob_data = blob_client.download_blob().readall()
        df = pd.read_csv(BytesIO(blob_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load CSV: {e}")

    if 'id' not in df.columns or 'mission' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'id' and 'mission' columns")

    df = df.rename(columns={'id': 'kicid'})

    # Extract features
    feature_df = create_feature_dataset_in_batches(df, batch_size=5)

    try:
        if is_ensemble:
            required_models = ["xgb", "lgbm", "catboost", "nn"]
            missing_models = [m for m in required_models if m not in models]
            if missing_models:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing models for ensemble: {missing_models}"
                )

            # Collect probabilities from all models
            all_probas = []
            for m in required_models:
                model = models[m]
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(feature_df)[:, 1]
                else:
                    # fallback for models that don't support predict_proba
                    probas = model.predict(feature_df)
                all_probas.append(probas)

            # Stack and average
            avg_probas = np.mean(np.column_stack(all_probas), axis=1)
            preds = (avg_probas >= 0.5).astype(int)

            feature_df["prediction"] = preds
            feature_df["prediction_proba"] = avg_probas

        else:
            model = models[model_type]
            preds = model.predict(feature_df)
            probas = (
                model.predict_proba(feature_df)[:, 1]
                if hasattr(model, "predict_proba")
                else [None] * len(preds)
            )
            feature_df["prediction"] = preds
            feature_df["prediction_proba"] = probas

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    results = convert_numpy_types(feature_df.to_dict(orient="records"))
    return {"results": results}