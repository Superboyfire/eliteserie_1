import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from features.build_features import prepare_data

data = pd.read_csv("data/eliteserien_data.csv")
X, y = prepare_data(data)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "model/rf_models.pkl")
