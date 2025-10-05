import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os


from azure.storage.blob import BlobServiceClient


def load_train():

    account_url = os.getenv("AZURE_ACCOUNT_URL")
    sas_token = os.getenv("AZURE_SAS_TOKEN")
    container_name = "artifacts"
    blob_path = "data/train_feature_df.csv"

    if not account_url or not sas_token:
        raise ValueError("Missing AZURE_ACCOUNT_URL or AZURE_SAS_TOKEN environment variable.")

    # Construct full blob URL with SAS token
    blob_url = f"{account_url}/{container_name}/{blob_path}?{sas_token}"

    # Load CSV using pandas
    df = pd.read_csv(blob_url)
    return df

# -----------------------------
# LGBM - Handles NaNs
# -----------------------------
def train_lgbm(params, X, y): 
    model = lgb.LGBMClassifier( **params ) 
    model.fit(X, y) 
    return model

# -----------------------------
# XGBoost - Handles NaNs
# -----------------------------
def train_xgb(params, X, y):
    model = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X, y)
    return model

# -----------------------------
# CatBoost - Handles NaNs natively
# -----------------------------
def train_catboost(params, X, y):
    model = CatBoostClassifier(
        **params,
    )
    model.fit(X, y)
    return model

# -----------------------------
# Neural Network (Keras) - Needs preprocessing
# -----------------------------
def train_nn(params, X, y):
    # Replace NaNs with column means (or use more advanced imputation)
    X = X.copy()
    X.fillna(X.mean(), inplace=True)

    # Scale inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]

    model = Sequential()
    model.add(Dense(params.get("units1", 64), activation='relu', input_dim=input_dim))
    model.add(Dropout(params.get("dropout1", 0.2)))
    model.add(Dense(params.get("units2", 32), activation='relu'))
    model.add(Dropout(params.get("dropout2", 0.2)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=params.get("learning_rate", 1e-3)),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_scaled, y,
        epochs=params.get("epochs", 20),
        batch_size=params.get("batch_size", 32),
        verbose=1
    )

    return model, scaler  # return scaler if you want to transform test data later