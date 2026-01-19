import time
import random
import threading
from store.redis_client import FeatureStore

fs = FeatureStore()

def simulate_transactions():
    """Simulates calculating rolling window features from a transaction stream"""
    print("🔄 Starting Feature Engineering Worker...")
    
    users = ["user_101", "user_102", "user_103"]
    
    while True:
        # Simulate a new transaction coming in
        user = random.choice(users)
        amount = random.uniform(10, 500)
        
        # 1. Compute 'Real-Time' Features 
        # (In reality, this logic would process a sliding window of events)
        # Feature: Last transaction amount
        fs.set_feature(user, "last_txn_amt", amount)
        
        # Feature: Count of transactions in last hour (incrementing mock)
        current_count = fs.get_features(user, ["txn_count_1h"])[0]
        fs.set_feature(user, "txn_count_1h", current_count + 1)
        
        print(f"Updated features for {user}: Amt=${amount:.2f}, Count={current_count+1}")
        time.sleep(2)

if __name__ == "__main__":
    simulate_transactions()
