import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="📦 Supply Chain Demand & Inventory Optimizer",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #0f172a; }
    .sub-title { font-size: 1.05rem; color: #475569; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📦 Supply Chain Demand & Inventory Buffer Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Developed by <b>Arjuna Fransesco</b> | Machine Learning & Supply Chain Analytics Portfolio</div>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "models", "lightgbm_demand_forecaster.joblib"))
    features = joblib.load(os.path.join(base_dir, "models", "feature_columns.joblib"))
    with open(os.path.join(base_dir, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    return model, features, metrics

model, features, metrics = load_assets()

# Sidebar
st.sidebar.header("🎛️ Inventory & Demand Simulation")
sku_choice = st.sidebar.selectbox("Select SKU", ["SKU-ELECTRONICS-01", "SKU-BEVERAGE-02", "SKU-APPAREL-03", "SKU-FMCG-04"])
unit_price = st.sidebar.slider("Unit Price ($)", 10.0, 150.0, 45.0, step=1.0)
is_promo = st.sidebar.checkbox("Active Marketing Promotion?", value=False)
lead_time = st.sidebar.slider("Supplier Lead Time (Days)", 1, 14, 5)
service_level = st.sidebar.selectbox("Target Service Level", ["90% (Z=1.28)", "95% (Z=1.65)", "99% (Z=2.33)"], index=1)
recent_sales_lag = st.sidebar.number_input("Yesterday's Sales (Units)", min_value=0, max_value=500, value=85)

z_map = {"90% (Z=1.28)": 1.28, "95% (Z=1.65)": 1.645, "99% (Z=2.33)": 2.33}
z_val = z_map[service_level]

# Feature DataFrame
sample_input = pd.DataFrame([{
    "unit_price": unit_price,
    "is_promotion": 1 if is_promo else 0,
    "is_weekend": 0,
    "lead_time_days": lead_time,
    "dayofweek": 2,
    "month": 6,
    "dayofyear": 170,
    "lag_1d": recent_sales_lag,
    "lag_7d": recent_sales_lag * 0.95,
    "lag_14d": recent_sales_lag * 1.02,
    "rolling_mean_7d": recent_sales_lag * 0.98,
    "rolling_std_7d": 12.5
}])[features]

predicted_demand = max(0, int(np.round(model.predict(sample_input)[0])))
safety_stock = int(np.ceil(z_val * 18.5 * np.sqrt(lead_time)))
reorder_point = int(np.ceil(predicted_demand * lead_time + safety_stock))

# KPI Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Forecasted Daily Demand", f"{predicted_demand} Units")
with c2:
    st.metric("Optimal Safety Stock", f"{safety_stock} Units")
with c3:
    st.metric("Reorder Point (ROP)", f"{reorder_point} Units")
with c4:
    st.metric("Model WAPE Error", f"{metrics['LightGBM_Demand_Forecaster']['WAPE_Percent']}%", delta="R² = " + str(metrics['LightGBM_Demand_Forecaster']['R2_Score']))

st.markdown("---")

tab1, tab2 = st.tabs(["📊 Forecast Evaluation & Feature Importance", "📑 System Specifications"])
with tab1:
    st.image("reports/demand_forecast_evaluation.png", use_container_width=True)

with tab2:
    st.json(metrics)
