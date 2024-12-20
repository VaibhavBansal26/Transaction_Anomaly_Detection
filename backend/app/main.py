from typing import Union

from fastapi import FastAPI
# import dill
from pydantic import BaseModel
import pandas as pd
# import numpy as np

import numpy as np
import joblib
app = FastAPI()

import dill
with open('./app/rfr_v1.pkl', 'rb') as f:
    reloaded_model = joblib.load(f)


class Payload(BaseModel):
    Transaction_Amount: float
    Average_Transaction_Amount: float
    Frequency_of_Transactions: float

app = FastAPI()


@app.get("/")
def read_root():
    return {
        "Name": "Vaibhav Bansal",
        "Project": "Transaction Anomaly Detection",
        "University": "SUNY Buffalo"
    }


@app.post("/predict")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    y_hat = reloaded_model.predict(df)
    return {"prediction": y_hat[0]}