# ⚡ Real-Time ML Feature Store

**Project Goal:** Build a low-latency Feature Store for fraud detection, separating feature computation (Write path) from inference (Read path).

## 🏗 Architecture
1.  **Feature Store:** Redis (Key-Value) for sub-millisecond access.
2.  **Write Path:** `feature_eng/compute_features.py` continuously updates user profiles based on stream (simulated).
3.  **Read Path:** `serving/inference_api.py` (FastAPI) fetches features at runtime to score transactions.

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Feature Engineering Worker
This script simulates the background process extracting features updates.
```bash
python feature_eng/compute_features.py
```

### 3. Start Inference API
Open a new terminal. This is the endpoint the payment gateway would call.
```bash
uvicorn serving.inference_api:app --reload
```

### 4. Test Prediction
Send a POST request to `http://127.0.0.1:8000/predict`:
```json
{
  "user_id": "user_101",
  "amount": 5000.00,
  "merchant_id": "M-2023"
}
```

## 🛠 Tech Stack
- **FastAPI**: REST Interface
- **Redis**: Feature Store
- **Python**: Logic and Simulation
