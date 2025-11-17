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
    st.error("No data loaded. Upload a CSV using the sidebar or place default dataset at the path shown in the sidebar.")
    st.stop()

# ---------------------------
# Navigation
# ---------------------------
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "Home",
    "Data Overview",
    "Classification",
    "Clustering",
    "Association Rules (ARM)",
    "Regression",
    "Dynamic Pricing",
    "Persona Generator",
    "Insights"
])

# ---------------------------
# HOME
# ---------------------------
if page == "Home":
    st.title("AI–VR Fashion Intelligence Suite")
    st.markdown("""
    This dashboard bundles ML models (classification, clustering, regression), association rule mining,
    a dynamic pricing simulator, persona generation and downloadable outputs — all in one Streamlit app.
    """)
    st.markdown("**Dataset:**")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    if st.button("Preview dataset (first 10 rows)"):
        st.dataframe(df.head(10))
    st.markdown("---")
    st.markdown("**Quick start:** use **Data Overview** to inspect data, then choose an analysis tab.")
    st.markdown("---")

# ---------------------------
# DATA OVERVIEW
# ---------------------------
elif page == "Data Overview":
    st.title("Data Overview")
    st.subheader("Preview")
    st.dataframe(df.head(10))
    st.subheader("Shape & Types")
    st.write(df.shape)
    st.write(df.dtypes)
    st.subheader("Missing values")
    st.write(df.isnull().sum().sort_values(ascending=False).head(30))
    st.subheader("Summary statistics (numeric)")
    st.write(df.describe().T)
    st.markdown("---")
    download_df(df, "data_overview_download.csv")

# ---------------------------
# CLASSIFICATION
# ---------------------------
elif page == "Classification":
    st.title("Classification")
    st.markdown("Select columns and model to build a classifier.")

    # select target and features
    target = st.selectbox("Target column (y)", df.columns)
    possible_features = [c for c in df.columns if c != target]
    features = st.multiselect("Feature columns (X) — if none selected, numerical columns will be used", possible_features,
                              default=[c for c in df.select_dtypes(include=np.number).columns if c != target])

    if not features:
        st.warning("No features selected — please choose at least one feature.")
    else:
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest", "XGBoost (if available)"])
        test_size = st.slider("Test size (%)", 10, 50, 20) / 100.0

        if st.button("Train classifier"):
            # Prepare dataset
            X = df[features].copy()
            y = df[target].copy()

            # Basic preprocessing: drop NA rows for simplicity
            data = pd.concat([X, y], axis=1).dropna()
            X = pd.get_dummies(data[features])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Model selection
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=200, random_state=42)
            else:
                if _HAS_XGB:
                    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
                else:
                    st.warning("XGBoost not available — falling back to Random Forest")
                    model = RandomForestClassifier(n_estimators=200, random_state=42)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Metrics
            st.subheader("Metrics")
            st.write("Accuracy:", accuracy_score(y_test, preds))
            st.write("Precision:", precision_score(y_test, preds, average="weighted", zero_division=0))
            st.write("Recall:", recall_score(y_test, preds, average="weighted", zero_division=0))
            st.write("F1 score:", f1_score(y_test, preds, average="weighted", zero_division=0))

            st.subheader("Confusion matrix")
            cm = confusion_matrix(y_test, preds)
            st.write(cm)

            # Show a small results table and download
            results = pd.DataFrame({"Actual": y_test, "Predicted": preds}).reset_index(drop=True)
            st.dataframe(results.head(50))
            download_df(results, "classification_results.csv")

# ---------------------------
# CLUSTERING
# ---------------------------
elif page == "Clustering":
    st.title("Clustering")
    st.markdown("Choose numeric features and clustering algorithm.")

    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    if not numeric_cols:
        st.error("No numeric features available for clustering.")
    else:
        cluster_features = st.multiselect("Numeric features for clustering", numeric_cols, default=numeric_cols[:4])
        n_clusters = st.slider("Number of clusters (k)", 2, 12, 3)
        method = st.selectbox("Method", ["KMeans", "Agglomerative"])

        if st.button("Run clustering"):
            X = df[cluster_features].dropna()
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            if method == "KMeans":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(Xs)
            else:
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(Xs)

            df_clusters = df.loc[X.index].copy()
            df_clusters["cluster"] = labels

            st.subheader("Cluster counts")
            st.write(df_clusters["cluster"].value_counts().sort_index())

            # PCA 2D
            pca = PCA(n_components=2)
            comp = pca.fit_transform(Xs)
            df_clusters["pc1"], df_clusters["pc2"] = comp[:, 0], comp[:, 1]
            fig = px.scatter(df_clusters, x="pc1", y="pc2", color="cluster", hover_data=df_clusters.columns)
            st.plotly_chart(fig, use_container_width=True)

            download_df(df_clusters, "clustered_dataset.csv")

# ---------------------------
# ASSOCIATION RULES (ARM)
# ---------------------------
elif page == "Association Rules (ARM)":
    st.title("Association Rule Mining (Apriori)")
    st.markdown("Select categorical columns (discrete choices) for basket-style analysis.")

    cat_cols = st.multiselect("Categorical columns", [c for c in df.columns if df[c].dtype == "object"], default=[c for c in df.columns if df[c].dtype == "object"][:6])
    min_support = st.slider("Min support", 0.01, 0.5, 0.05)
    min_confidence = st.slider("Min confidence", 0.1, 1.0, 0.5)
    min_lift = st.slider("Min lift", 0.5, 5.0, 1.0)

    if st.button("Run Apriori"):
        if not cat_cols:
            st.error("Pick at least one categorical column.")
        else:
            # One-hot encoding for apriori
            df_cat = df[cat_cols].fillna("NA").astype(str)
            df_hot = pd.get_dummies(df_cat)

            freq = apriori(df_hot, min_support=min_support, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
            rules = rules[rules["lift"] >= min_lift].sort_values(["confidence", "lift"], ascending=False)

            if rules.empty:
                st.warning("No rules found for the given thresholds.")
            else:
                st.subheader("Rules")
                st.dataframe(rules)
                download_df(rules, "association_rules.csv")

# ---------------------------
# REGRESSION
# ---------------------------
elif page == "Regression":
    st.title("Regression")
    st.markdown("Select numeric target and features for regression models.")

    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    if not numeric_cols:
        st.error("No numeric columns available for regression.")
    else:
        target = st.selectbox("Target (y)", numeric_cols)
        features = st.multiselect("Features (X)", [c for c in numeric_cols if c != target], default=[c for c in numeric_cols if c != target][:5])

        model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor"])
        test_size = st.slider("Test size (%)", 10, 50, 20) / 100.0

        if st.button("Train regression"):
            X = df[features].dropna()
            y = df.loc[X.index, target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest Regressor":
                model = RandomForestRegressor(n_estimators=300, random_state=42)
            elif model_choice == "Gradient Boosting Regressor":
                model = GradientBoostingRegressor(random_state=42)
            else:
                if _HAS_XGB:
                    model = XGBRegressor(objective="reg:squarederror", random_state=42)
                else:
                    st.warning("XGBoost not available — falling back to Random Forest")
                    model = RandomForestRegressor(n_estimators=300, random_state=42)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.subheader("Metrics")
            st.write("R²:", r2_score(y_test, preds))
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
            st.write("MAE:", mean_absolute_error(y_test, preds))

            fig = px.scatter(x=y_test, y=preds, labels={"x": "Actual", "y": "Predicted"})
            st.plotly_chart(fig, use_container_width=True)

            out = pd.DataFrame({"Actual": y_test, "Predicted": preds})
            download_df(out, "regression_predictions.csv")

# ---------------------------
# DYNAMIC PRICING
# ---------------------------
elif page == "Dynamic Pricing":
    st.title("Dynamic Pricing Simulator")

    st.markdown("Adjust cost, margins and persona multipliers to see recommended price and revenue projections.")

    cost = st.slider("Cost ($)", 5, 500, 60)
    margin_pct = st.slider("Base margin (%)", 1, 200, 40)
    persona_adj_pct = st.slider("Persona multiplier (%)", 50, 200, 100)
    feature_addon = st.slider("Feature add-on ($)", 0, 300, 50)

    price = cost * (1 + margin_pct / 100.0) * (persona_adj_pct / 100.0) + feature_addon
    st.success(f"Recommended price: ${price:0.2f}")

    units = st.slider("Simulate units sold", 10, 1000, 100)
    revenue = price * units
    st.write(f"Projected revenue for {units} units: ${revenue:0.2f}")

    # sensitivity chart
    price_range = np.linspace(max(1, cost*0.8), cost*3, 50)
    revenue_curve = price_range * units
    fig = px.line(x=price_range, y=revenue_curve, labels={"x": "Price", "y": "Revenue"})
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# PERSONA GENERATOR
# ---------------------------
elif page == "Persona Generator":
    st.title("Persona Generator")
    st.markdown("Use sliders to synthesize a persona and see the recommended marketing pitch.")

    fit_importance = st.slider("Fit importance (1-10)", 1, 10, 7)
    fit_difficulty = st.slider("Fit difficulty (1-10)", 1, 10, 6)
    tech_comfort = st.slider("Tech comfort (1-10)", 1, 10, 6)
    sustainability = st.slider("Sustainability preference (1-10)", 1, 10, 5)
    budget = st.slider("Budget sensitivity (1-10)", 1, 10, 5)

    # simple rule-based persona mapping
    if tech_comfort >= 8 and fit_importance >= 7:
        persona = "Metaverse Native"
        pitch = "Emphasize VR try-on, digital wardrobe, early-access perks."
    elif fit_importance >= 8 and fit_difficulty >= 7:
        persona = "Fit Frustrated"
        pitch = "Lead with Perfect-Fit Guarantee and made-to-order personalization."
    elif sustainability >= 8:
        persona = "Eco Warrior"
        pitch = "Highlight zero-waste production and eco-friendly materials."
    elif budget >= 8 or budget <= 3:
        persona = "Budget Conscious"
        pitch = "Promote value-based essentials and low-commitment try-before-you-buy."
    else:
        persona = "Premium Perfectionist"
        pitch = "Showcase premium fabrics, bespoke tailoring, and premium pricing options."

    st.markdown(f"### Persona: **{persona}**")
    st.write(pitch)
    st.markdown("---")

# ---------------------------
# INSIGHTS
# ---------------------------
elif page == "Insights":
    st.title("Insights & Recommendations")
    st.markdown("Auto-generated recommendations based on data and models.")

    # Quick computed signals (basic)
    if "InterestScore" in df.columns:
        top_persona = df.groupby("Persona")["InterestScore"].mean().idxmax() if "Persona" in df.columns else None
    else:
        top_persona = df["Persona"].mode()[0] if "Persona" in df.columns else None

    st.subheader("Key signals")
    if top_persona:
        st.write(f"- Highest interest persona (by average interest score): **{top_persona}**")
    st.write("- Top positive drivers to evaluate: Perfect Fit, Personalization, Sustainability (use Q17_... columns).")
    st.write("- Primary risk: price sensitivity among Budget Conscious segment.")

    st.markdown("### Suggested next steps")
    st.write("""
    1. Run clustering to identify highest-LTV segments.  
    2. Run regression to estimate WTP and set dynamic prices for suits/blazers.  
    3. Use classification to find likely early adopters for beta testing.  
    """)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>AI–VR Fashion Intelligence Suite • Built with Streamlit</p>", unsafe_allow_html=True)
