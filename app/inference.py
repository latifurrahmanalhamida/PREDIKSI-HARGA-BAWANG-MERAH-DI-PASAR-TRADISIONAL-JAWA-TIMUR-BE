import time
from typing import List, Tuple, Dict, Any
import numpy as np

def validate_window(window: List[float], expected_len: int) -> Tuple[bool, str]:
    if not isinstance(window, list):
        return False, "window must be a list"
    if len(window) != expected_len:
        return False, f"window must have length {expected_len}"
    try:
        _ = [float(x) for x in window]
    except Exception:
        return False, "window values must be numeric"
    return True, ""

def out_of_range_flags(window: List[float], scaler) -> List[str]:
    warnings = []
    if scaler is None:
        return warnings
    data_min = getattr(scaler, "data_min_", None)
    data_max = getattr(scaler, "data_max_", None)
    if data_min is not None and data_max is not None:
        min_v = float(data_min[0])
        max_v = float(data_max[0])
        for v in window:
            if v < min_v or v > max_v:
                warnings.append("out_of_range")
                break
    return warnings

def predict_next(window: List[float], scaler, model, window_size: int) -> Tuple[float, Dict[str, Any]]:
    t0 = time.time()
    arr = np.array(window, dtype=float).reshape(-1, 1)
    arr_scaled = scaler.transform(arr)
    X_input = arr_scaled.reshape(1, window_size, 1)
    y_scaled = model.predict(X_input, verbose=0)
    y_real = scaler.inverse_transform(y_scaled).ravel()[0]
    runtime_ms = (time.time() - t0) * 1000.0
    return float(y_real), {"runtime_ms": runtime_ms}