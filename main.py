from fastapi import FastAPI,HTTPException,Query,Path,Depends
from fastapi.responses import JSONResponse
from Schema.schema import CarFeatures, PredictionPrice
from Model.model import load_artifacts, predict_price

app = FastAPI(title="Car Price Prediction Model")

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/")
def message():
    return JSONResponse(
        status_code=200,
        content={"success" : True,
                 "detail":"This is the test run for this API"}
    )

@app.post("/predict",response_model=PredictionPrice)
def prdict_price(features  : CarFeatures):
    price = predict_price(features.model_dump())
    return PredictionPrice(prediction_price=price)
