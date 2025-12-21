import os
import json
import time
import pickle
from typing import Dict, Any
from tensorflow.keras.models import load_model

class ModelEntry:
    def __init__(self, region: str, model_path: str, scaler_path: str, trained_until: str, window_size: int, meta: Dict[str, Any]):
        self.region = region
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.trained_until = trained_until
        self.window_size = window_size
        self.meta = meta
        self.model = None
        self.scaler = None

    def load(self):
        t0 = time.time()
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.model = load_model(self.model_path)
        return (time.time() - t0) * 1000.0  # ms

def load_registry(base_dir: str) -> Dict[str, ModelEntry]:
    registry_path = os.path.join(os.path.dirname(__file__), "models_registry.json")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"models_registry.json not found at {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries: Dict[str, ModelEntry] = {}
    for region, cfg in data.items():
        if not cfg.get("is_active", False):
            continue
        model_path = cfg["path_model"]
        scaler_path = cfg["path_scaler"]
        trained_until = cfg.get("trained_until", "2024-04-30")
        window_size = int(cfg.get("window_size", 7))
        meta = {
            "mape_2024": cfg.get("mape_2024"),
            "mape_2025": cfg.get("mape_2025"),
        }
        if base_dir and not os.path.isabs(model_path):
            model_path = os.path.join(base_dir, model_path)
        if base_dir and not os.path.isabs(scaler_path):
            scaler_path = os.path.join(base_dir, scaler_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found for {region}: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found for {region}: {scaler_path}")

        entries[region] = ModelEntry(region, model_path, scaler_path, trained_until, window_size, meta)

    return entries