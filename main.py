from typing import List
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import predict

app = FastAPI()


class InputIn(BaseModel):
    input_data: List[int]


class Output(BaseModel):
    prediction: int


@app.get("/ping")
def pong():
    return {"ping": "pong!"}


@app.post("/predict", response_model=Output, status_code=200)
def get_prediction(payload: InputIn):
    inp = payload.input_data

    prediction = predict(inp)

    if not prediction:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {"input_data": inp, "prediction": prediction}
    return response_object
