import os
import json
import joblib

# --- Paths (adjust if needed) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "housing_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.json")


try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model at {MODEL_PATH}: {e}")

try:
    with open(FEATURES_PATH, "r") as f:
        feature_info = json.load(f)
        FEATURE_ORDER = feature_info.get("feature_order") or feature_info.get("feature_names")
        if not FEATURE_ORDER:
            raise ValueError("feature_order or feature_names key not found in model_features.json")
except Exception as e:
    raise RuntimeError(f"Could not load feature metadata: {e}")