import os
os.makedirs("model", exist_ok=True)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from features.build_features import prepare_data

# Last inn data
data = pd.read_csv("data/eliteserien_data.csv")

# Forbered data (X, y)
X, y = prepare_data(data)  # Pass på at denne funksjonen returnerer riktig!

# Tren modellen
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Lagre modellen i riktig mappe
joblib.dump(model, "models/rf_model.pkl")
print("Modellen er lagret i model/rf_model.pkl")
