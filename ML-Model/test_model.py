import pandas as pd
import joblib

# Load model
model = joblib.load("anomaly_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
data = pd.read_csv("self_healing_dataset.csv")

features = data[[
    "cpu_usage",
    "memory_usage",
    "request_rate",
    "latency",
    "pod_restarts"
]]

X_scaled = scaler.transform(features)

predictions = model.predict(X_scaled)

# -1 = anomaly, 1 = normal
data["prediction"] = predictions

print(data["prediction"].value_counts())
