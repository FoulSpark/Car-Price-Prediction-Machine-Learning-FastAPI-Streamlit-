from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "Dataset"

data = pd.read_csv(DATASET_DIR / "cardekho_data (1).csv")

data = data.drop(columns=["Car_Name"])

X = data.drop(columns=["Selling_Price"])
y = data["Selling_Price"]

categorical_columns = ["Fuel_Type", "Seller_Type", "Transmission"]

ct = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

X_transformed = ct.fit_transform(X)

model = RandomForestRegressor(random_state=42)
model.fit(X_transformed, y)

joblib.dump(model, DATASET_DIR / "random_forest_model.pkl")
joblib.dump(ct, DATASET_DIR / "transformer.pkl")

print("Model and transformer saved.")