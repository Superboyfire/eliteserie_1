import requests
import pandas as pd

def fetch_eliteserien_data():
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": "DIN_API_NOKKEL"}
    params = {"league": "103", "season": "2024"}
    res = requests.get(url, headers=headers, params=params)
    fixtures = res.json()["response"]

    rows = []
    for f in fixtures:
        row = {
            "match": f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}",
            "home_team": f['teams']['home']['name'],
            "away_team": f['teams']['away']['name'],
            "home_goals": f['goals']['home'],
            "away_goals": f['goals']['away'],
            "date": f['fixture']['date']
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("data/eliteserien_data.csv", index=False)
