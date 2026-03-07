from pathlib import Path
import pandas as pd
import joblib

CAR_PRICE_API_DIR = Path(__file__).resolve().parent
DATASET_DIR = CAR_PRICE_API_DIR.parent / "Dataset"
MODEL_PATH = DATASET_DIR / "random_forest_model.pkl"
TRANSFORMER_PATH = DATASET_DIR / "transformer.pkl"

_model = None
_transformer = None


def load_artifacts():
    global _model, _transformer

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    if _transformer is None:
        _transformer = joblib.load(TRANSFORMER_PATH)


def preprocess(payload: dict):
    df = pd.DataFrame([payload])
    X = _transformer.transform(df)
    return X


def predict_price(payload: dict) -> float:
    load_artifacts()
    X = preprocess(payload)
    pred = _model.predict(X)[0]
    return float(pred)