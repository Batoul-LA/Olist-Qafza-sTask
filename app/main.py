from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sys
import os
request_count = {"predict": 0, "predict_batch": 0}

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict, load_artifacts

app = FastAPI(title="Olist Late Delivery Prediction API")

model, feature_columns, config = load_artifacts()


class Order(BaseModel):
    order_purchase_timestamp: str
    num_items: float
    total_price: float
    total_freight: float
    total_payment_value: float
    num_payment_methods: float
    customer_state: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str = "final_model_v1"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_type": "LogisticRegression",
        "model_version": "final_model_v1",
        "threshold": config["model"]["threshold"],
        "n_features": len(feature_columns)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(order: Order):
    try:
        request_count["predict"] += 1
        result = predict(order.model_dump(), model=model, feature_columns=feature_columns, cfg=config)
        return result[0]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch(orders: List[Order]):
    try:
        request_count["predict_batch"] += 1
        results = []
        for order in orders:
            r = predict(order.model_dump(), model=model, feature_columns=feature_columns, cfg=config)
            results.append(r[0])
        return results
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal prediction error")

    @app.get("/metrics")
    def metrics():
        return {"total_predict_requests": request_count["predict"],
            "total_batch_requests": request_count["predict_batch"]}