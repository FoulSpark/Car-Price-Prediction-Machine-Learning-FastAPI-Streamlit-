# Car Price Prediction (Machine Learning + FastAPI + Streamlit)

A complete end-to-end **car selling price prediction** project built using **Machine Learning (Random Forest Regressor)** and deployed behind a **FastAPI** inference API, with an optional **Streamlit** UI.

The model is trained on the popular **CarDekho** dataset and predicts the expected **selling price (in lakhs)** based on key vehicle attributes.

## Project Highlights

- **Model**: `RandomForestRegressor` (scikit-learn)
- **Preprocessing**: `ColumnTransformer` + `OneHotEncoder` for categorical variables
- **Serving**: FastAPI `/predict` endpoint
- **UI**: Streamlit app that calls the API
- **Performance**: ~**98% accuracy** (see the **Model Performance** section below)

## Dataset

This project uses the CarDekho dataset stored locally as:

- `cardekho_data (1).csv`

### Target

- `Selling_Price` (the value the model learns to predict)

### Features used for training

The training script drops `Car_Name` and trains using:

- `Year`
- `Present_Price` (in lakhs)
- `Kms_Driven`
- `Fuel_Type` (Petrol/Diesel/CNG)
- `Seller_Type` (Dealer/Individual)
- `Transmission` (Manual/Automatic)
- `Owner` (0/1/3, etc.)

## Model Training

Training is performed in `create_model.py`:

- Loads the CSV dataset
- Drops `Car_Name`
- One-hot encodes categorical columns (`Fuel_Type`, `Seller_Type`, `Transmission`)
- Trains a `RandomForestRegressor`
- Saves artifacts using `joblib`

Generated artifacts:

- `random_forest_model.pkl` (trained model)
- `transformer.pkl` (preprocessing transformer)

### Train the model locally

Run:

```bash
python create_model.py
```

This will overwrite the `random_forest_model.pkl` and `transformer.pkl` files with newly trained artifacts.

## Model Performance

In the experimentation notebook `model.ipynb`, the model is evaluated with a train/test split and reports a very strong score.

Typical metrics shown in the notebook include:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

The notebook reports an **R² Score ≈ 0.999**, which corresponds to approximately **98%+ accuracy** depending on how you choose to summarize regression performance.

> Note: for regression problems, “accuracy” is not a standard metric like it is in classification. Many projects use **R² score** (coefficient of determination) as an “accuracy-like” measure.

## FastAPI Inference API

The API is defined in `main.py` and uses:

- `schema.py` for request/response validation (Pydantic)
- `model.py` for artifact loading, preprocessing, and prediction

### Endpoints

- `GET /`
  - Simple health/test response
- `POST /predict`
  - Returns predicted selling price in lakhs

### Request schema

Example request payload:

```json
{
  "Year": 2014,
  "Present_Price": 5.59,
  "Kms_Driven": 27000,
  "Fuel_Type": "Petrol",
  "Seller_Type": "Dealer",
  "Transmission": "Manual",
  "Owner": 0
}
```

Example response:

```json
{
  "prediction_price": 3.42
}
```

### Run the API locally

Install dependencies (example):

```bash
pip install fastapi uvicorn scikit-learn pandas joblib pydantic
```

Start the server:

```bash
uvicorn main:app --reload
```

Then open:

- Swagger UI: `http://127.0.0.1:8000/docs`

## Streamlit UI (Optional)

A simple Streamlit UI is provided in `streamlit_app.py`. It collects user inputs and calls the FastAPI endpoint.

### Run Streamlit locally

Install dependencies:

```bash
pip install streamlit requests
```

Run:

```bash
streamlit run streamlit_app.py
```

### API URL configuration

In `streamlit_app.py`, set:

- `API_URL = "http://127.0.0.1:8000/predict"` for local development

A hosted endpoint is also included by default:

- `https://car-prediction-lpfl.onrender.com/predict`

## Project Structure

```text
.
├── cardekho_data (1).csv
├── create_model.py
├── main.py
├── model.py
├── schema.py
├── streamlit_app.py
├── model.ipynb
├── random_forest_model.pkl
├── transformer.pkl
└── feature_columns.pkl
```

## Notes / Gotchas

- The FastAPI input schema (`schema.py`) **does not include** `Car_Name`. The training script also drops it.
- `streamlit_app.py` currently shows an input for `Car_Name`, but the API schema expects the payload **without** it.
  - If you want the Streamlit UI to work with the current API, remove `Car_Name` from the Streamlit payload (or extend the API schema).

## Future Improvements

- Add a proper `requirements.txt`
- Add automated evaluation script for reproducible metrics
- Add model versioning and CI checks
- Add Docker support for easier deployment

