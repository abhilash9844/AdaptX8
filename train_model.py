"""Model Training"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
from pathlib import Path

def train_model():
    print("Training...")
    
    if not os.path.exists("dataset/dataset.csv"):
        print("No dataset! Run generate_dataset.py first.")
        return
    
    df = pd.read_csv("dataset/dataset.csv")
    
    X = df[['servers', 'workload', 'cpu', 'energy', 'temperature']].values
    y = df['efficiency'].values
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = RandomForestRegressor(
        n_estimators=150, max_depth=12,
        min_samples_split=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr_s, y_tr)
    
    pred = model.predict(X_te_s)
    print(f"R²: {r2_score(y_te, pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_te, pred):.4f}")
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/efficiency_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Model saved!")

if __name__ == "__main__":
    train_model()