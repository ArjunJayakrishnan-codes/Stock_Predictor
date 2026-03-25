"""
Local storage replacement for S3.
Models are saved to ./local_models/<SYMBOL>/ on disk.
"""
import joblib
import os
import json
from datetime import datetime

LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "local_models")


def _symbol_dir(symbol: str) -> str:
    path = os.path.join(LOCAL_MODELS_DIR, symbol)
    os.makedirs(path, exist_ok=True)
    return path


def upload_model(model, scaler, symbol: str):
    """Save model + scaler to local disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym_dir = _symbol_dir(symbol)

    model_path  = os.path.join(sym_dir, f"model_{timestamp}.pkl")
    scaler_path = os.path.join(sym_dir, f"scaler_{timestamp}.pkl")
    meta_path   = os.path.join(sym_dir, f"meta_{timestamp}.json")

    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "symbol":     symbol,
        "trained_at": timestamp,
        "model_path":  model_path,
        "scaler_path": scaler_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"[local] Model for {symbol} saved to {sym_dir}")


def download_model(symbol: str):
    """Load the latest saved model + scaler from disk."""
    sym_dir = _symbol_dir(symbol)
    files = os.listdir(sym_dir)

    model_files  = sorted([f for f in files if f.startswith("model_")])
    scaler_files = sorted([f for f in files if f.startswith("scaler_")])

    if not model_files or not scaler_files:
        raise FileNotFoundError(f"No saved model found for {symbol}")

    model  = joblib.load(os.path.join(sym_dir, model_files[-1]))
    scaler = joblib.load(os.path.join(sym_dir, scaler_files[-1]))
    return model, scaler


def list_saved_models():
    """List all model metadata from local disk."""
    models = []
    if not os.path.exists(LOCAL_MODELS_DIR):
        return models
    for symbol in os.listdir(LOCAL_MODELS_DIR):
        sym_dir = os.path.join(LOCAL_MODELS_DIR, symbol)
        if not os.path.isdir(sym_dir):
            continue
        for fname in os.listdir(sym_dir):
            if fname.startswith("meta_") and fname.endswith(".json"):
                with open(os.path.join(sym_dir, fname)) as f:
                    models.append(json.load(f))
    return sorted(models, key=lambda x: x["trained_at"], reverse=True)
