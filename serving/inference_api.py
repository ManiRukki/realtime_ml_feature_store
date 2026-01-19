from fastapi import FastAPI
from pydantic import BaseModel
from store.redis_client import FeatureStore
import random

app = FastAPI(title="Real-Time Fraud Detection API")
fs = FeatureStore()

class TransactionRequest(BaseModel):
    user_id: str
    amount: float
    merchant_id: str

@app.post("/predict")
def predict_fraud(txn: TransactionRequest):
    # 1. Fetch Real-Time Features from Store ( < 5ms latency guarantee )
    features = fs.get_features(txn.user_id, ["txn_count_1h", "last_txn_amt"])
    txn_count, last_amt = features
    
    # 2. Run Inference (Mocking a loaded ML model)
    # Logic: High velocity (count > 10) OR Sudden High Value (> 2x last) -> Fraud
    
    fraud_score = 0.0
    if txn_count > 10:
        fraud_score += 0.4
    if txn.amount > (last_amt * 2) and last_amt > 0:
        fraud_score += 0.5
        
    is_fraud = fraud_score > 0.5
    
    return {
        "user_id": txn.user_id,
        "is_fraud": is_fraud,
        "fraud_score": fraud_score,
        "features_used": {
            "txn_count_1h": txn_count,
            "last_txn_amt": last_amt,
            "current_amt": txn.amount
        }
    }

# To run: uvicorn serving.inference_api:app --reload
