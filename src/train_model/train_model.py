import numpy as np 
from sklearn.pipeline import Pipeline
import os
import io
from azure.storage.blob import BlobServiceClient
import joblib
from src.train_model.aux_train_model import *

def save_model_to_azure(pipeline, model_type):
    account_url = os.getenv("AZURE_ACCOUNT_URL")
    sas_token = os.getenv("AZURE_SAS_TOKEN")
    container_name = "artifacts"
    blob_path = f"models/{model_type}_model.joblib"

    if not account_url or not sas_token:
        raise ValueError("Missing AZURE_ACCOUNT_URL or AZURE_SAS_TOKEN environment variable.")

    # Create blob service and blob client
    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

    # Serialize the pipeline to bytes
    with io.BytesIO() as bytes_io:
        joblib.dump(pipeline, bytes_io)
        bytes_io.seek(0)
        blob_client.upload_blob(bytes_io, overwrite=True)

    print(f"Model saved to Azure Blob Storage at: {blob_path}")


def train_model(
        model_type, 
        params,
):
    
    train_set = load_train()
    train_set["target"] = np.where(train_set["koi_disposition"].isin(["CANDIDATE", "CONFIRMED"]), 1, 0)
    
    X = train_set.drop(
        ["kicid", "koi_disposition", "target", "Unnamed: 0"],
        axis=1
    )
    y = train_set["target"]
    
    if model_type == 'nn':
        model, scaler = train_nn(params, X, y)
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
    else:
        model = {
            'xgb': train_xgb,
            'catboost': train_catboost,
            'lgbm': train_lgbm
        }[model_type](params, X, y)
        pipeline = model

    save_model_to_azure(pipeline, model_type)

    return None