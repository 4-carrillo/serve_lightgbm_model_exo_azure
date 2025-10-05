import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

def load_train():
    # Example:
    # df = pd.read_csv("your_data.csv")
    # X = df.drop("target", axis=1)
    # y = df["target"]
    pass

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