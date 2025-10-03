import pandas as pd
from .model_loader import FEATURE_ORDER


# Helpful maps used in your training script
BINARY_COLUMNS = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
FURNISHING_MAP = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}

def _parse_value(feature, raw):
    """
    Converts incoming raw value to numeric according to feature.
    Accepts strings like 'yes'/'no', numeric strings, numeric values, and furnishing statuses.
    """
    if raw is None:
        raise ValueError(f"Missing feature: {feature}")

    # normalize
    if isinstance(raw, str):
        v = raw.strip().lower()
    else:
        v = raw

    # binary columns
    if feature in BINARY_COLUMNS:
        if isinstance(v, (int, float)):
            return 1 if float(v) != 0 else 0
        if v in ("yes", "y", "true", "1"):
            return 1
        if v in ("no", "n", "false", "0"):
            return 0
        raise ValueError(f"Invalid value for {feature}: {raw}")

    # furnishing status
    if feature == "furnishingstatus":
        if isinstance(v, (int, float)):
            return int(v)
        if v in FURNISHING_MAP:
            return FURNISHING_MAP[v]
        # allow numeric string
        if isinstance(v, str) and v.isdigit():
            return int(v)
        raise ValueError(f"Invalid furnishingstatus: {raw}")

    # numeric features
    try:
        return float(v)
    except Exception:
        raise ValueError(f"Invalid numeric value for {feature}: {raw}")

def preprocess_input(payload: dict):
    row = []
    for feat in FEATURE_ORDER:
        val = _parse_value(feat, payload[feat])
        row.append(val)
    df = pd.DataFrame([row], columns=FEATURE_ORDER)  # DataFrame instead of np.array
    return df