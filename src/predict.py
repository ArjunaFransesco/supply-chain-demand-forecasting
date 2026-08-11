import joblib
import pandas as pd
import os

def forecast_demand(feature_dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base_dir, "models", "lightgbm_demand_forecaster.joblib"))
    features = joblib.load(os.path.join(base_dir, "models", "feature_columns.joblib"))
    df = pd.DataFrame([feature_dict])[features]
    pred = model.predict(df)[0]
    return {"forecasted_sales_units": max(0, int(round(pred)))}
