"""
AI–VR Fashion Intelligence Suite
Single-file Streamlit app with:
- Data upload + local fallback
- Classification (LR, RF, XGBoost)
- Clustering (KMeans, Agglomerative)
- Association Rules (Apriori)
- Regression (LR, RF, GBR, XGBoost)
- Dynamic pricing simulator
- Persona generator
- Downloadable outputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

# XGBoost (optional, in requirements)
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ---------------------------
# Styling (premium gradient)
# ---------------------------
PAGE_STYLE = """
<style>
body {
  background: linear-gradient(135deg,#0f172a 0%, #0b1220 50%, #071021 100%);
  color: #f1f5f9;
}
.stButton>button {
  background-color: #7c3aed;
  color: white;
}
.sidebar .sidebar-content {
  background: linear-gradient(180deg,#061026,#071836);
  color: #e6eef8;
}
h1, h2, h3, h4 { color: #e6eef8; }
</style>
"""
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()

def download_df(df: pd.DataFrame, filename: str):
    st.download_button(label=f"Download {filename}", data=to_csv_bytes(df), file_name=filename, mime="text/csv")

# ---------------------------
# Data loader
# ---------------------------
st.sidebar.header("Data Loader")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

# local fallback path (user said dataset is already downloaded). Update if your path differs.
local_default_path = "/mnt/data/AI_VR_Fashion_Survey_Synthetic_Data (1).csv"

@st.cache_data
def load_default():
    # Try uploaded first (handled outside), then local path, then try a neutral empty df
    try:
        df_local = pd.read_csv(local_default_path)
        return df_local
    except Exception:
        return pd.DataFrame()

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_default()
    if df.empty:
        st.sidebar.warning("No local dataset found. Please upload CSV or place dataset at:\n" + local_default_path)

# stop if still empty
if df is None or df.empty:
    st.title("AI–VR Fashion Intelligence Suite")
    st.error("No data loaded. Upload a CSV using the sidebar or place default dataset
