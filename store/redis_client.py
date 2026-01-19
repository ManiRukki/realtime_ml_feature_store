import redis
import time
import random

# Connect to Redis (Mocked for script availability check)
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    REDIS_AVAILABLE = True
except:
    print("⚠️ Redis not available. Running in Mock Mode.")
    REDIS_AVAILABLE = False

class FeatureStore:
    def __init__(self):
        self.redis = r if REDIS_AVAILABLE else None
        self.mock_store = {}

    def set_feature(self, entity_id, feature_name, value, ttl=3600):
        key = f"{entity_id}:{feature_name}"
        if self.redis:
            self.redis.setex(key, ttl, value)
        else:
            self.mock_store[key] = value

    def get_features(self, entity_id, feature_names):
        """Batch retrieve features for low-latency inference"""
        values = []
        for name in feature_names:
            key = f"{entity_id}:{name}"
            if self.redis:
                val = self.redis.get(key)
                values.append(float(val) if val else 0.0)
            else:
                values.append(self.mock_store.get(key, 0.0))
        return values

if __name__ == "__main__":
    # Test
    fs = FeatureStore()
    fs.set_feature("user_123", "avg_spend_7d", 150.50)
    print(f"Retrieved: {fs.get_features('user_123', ['avg_spend_7d'])}")
