from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class CoinRequest(BaseModel):
    coin: str
    week: int

@app.post("/predict")
def predict(request: CoinRequest):
    try:
        coin = request.coin.lower()
        week = request.week
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=365&interval=daily&precision=5"
        headers = {"accept": "application/json"}

        day = 365 + (week * 7)

        response = requests.get(url, headers=headers)
        K = json.loads(response.text)
        Y = K["prices"]

        DataTable = pd.DataFrame(Y, columns=["Market Cap", "Price"])
        DataTable.index = pd.Index(range(1, len(DataTable) + 1))
        DataTable.insert(0, "#", range(1, len(DataTable) + 1))

        X = DataTable[["#"]]
        Y = DataTable["Price"]

        linear_model = LinearRegression()
        linear_model.fit(X, Y)
        linear_prediction = max(0, linear_model.predict([[day]])[0])

        degree = 2
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        poly_model.fit(X, Y)
        poly_prediction = max(0, poly_model.predict([[day]])[0])

        high = DataTable["Price"].max()
        low = DataTable["Price"].min()
        avg = DataTable["Price"].mean()
        curr = DataTable["Price"].iloc[-1]

        exp_low = min(linear_prediction, poly_prediction)
        exp_high = max(linear_prediction, poly_prediction)
        exp_avg = (exp_high + exp_low) / 2

        result = {
            "365 Days High": round(high, 2),
            "365 Days Low": round(low, 2),
            "365 Days Average": round(avg, 2),
            "Current Price": round(curr, 2),
            "Predicted High Price": round(exp_high, 1),
            "Predicted Average Price": round(exp_avg, 1),
            "Predicted Low Price": round(exp_low, 1)
        }

        return result
    except Exception as e:
        return {"error": str(e)}
