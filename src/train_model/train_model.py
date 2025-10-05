import pandas as pd
import numpy as np
from aux_train_model import *


def train_model(
        model_type, 
        params, 
        azure_upload=False,
):
    
    train_set = load_train()
    train_set["target"] = np.where(train_set["koi_disposition"].isin(["CANDIDATE", "CONFIRMED"]), 1, 0)
    
    X = train_set.drop(
        ["kicid", "koi_disposition", "target", "Unnamed: 0"],
        axis=1
    )
    y = train_set["target"]
    
    # Train model
    if model_type == 'xgb':
        model = train_xgb(params, X, y)

    elif model_type == 'catboost':
        model = train_catboost(params, X, y)

    elif model_type == 'lgbm':
        model = train_lgbm(params, X, y)

    elif model_type == 'nn':
        model, scaler = train_nn(params, X, y)
        # Save both model and scaler if needed
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save and upload to Azure
    if azure_upload:
        # Create temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, f"{model_name or model_type}_model.pkl")

            # Save model using joblib (or keras saving for NN)
            if model_type == 'nn':
                model_path = os.path.join(tmpdir, f"{model_name or model_type}_model.keras")
                model.save(model_path)  # save keras model
            else:
                joblib.dump(model, model_path)

            # Upload
            upload_to_azure_blob(
                local_file_path=model_path,
                container_name=container_name,
                blob_name=os.path.basename(model_path),
                connection_string=connection_string
            )

    return (model, scaler) if model_type == 'nn' else model