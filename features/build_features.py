import pandas as pd

def build_features(data, match):
    row = data[data["match"] == match].iloc[0]
    return [
        row["home_goals"],
        row["away_goals"]
    ]

def prepare_data(data):
    X = data[["home_goals", "away_goals"]]
    y = (data["home_goals"] > data["away_goals"]).astype(int)  # 1 hvis hjemmelag vinner
    return X, y
