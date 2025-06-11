import streamlit as st
import joblib
import pandas as pd
from features.build_features import build_features

st.title("Eliteserien AI-prediksjon")

model = joblib.load("./model/rf_model.pkl")
data = pd.read_csv("data/eliteserien_data.csv")

match = st.selectbox("Velg kamp", data["match"])
features = build_features(data, match)
pred_winner = model.predict([features])[0]
pred_proba = model.predict_proba([features])[0]

st.subheader("Prediksjon")
st.write(f"🔮 Vinner: {'Hjemmelag' if pred_winner == 1 else 'Bortelag'}")
st.write(f"🏆 Sjanser - Hjemme: {round(pred_proba[1]*100, 1)}% | Borte: {round(pred_proba[0]*100, 1)}%")
