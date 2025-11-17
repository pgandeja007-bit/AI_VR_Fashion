##############################################
#        AI–VR FASHION INTELLIGENCE SUITE
#                FINAL app.py
##############################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt

##############################################
# PREMIUM GRADIENT THEME CSS
##############################################
GRADIENT_STYLE = """
<style>
body {
    background: linear-gradient(135deg, #1c1c29 0%, #202040 50%, #1a1a2e 100%);
    color: white !important;
}
.sidebar .sidebar-content {
    background-color: #1b1b2f !important;
}
</style>
"""

st.markdown(GRADIENT_STYLE, unsafe_allow_html=True)

##############################################
# DOWNLOAD BUTTON UTILITY
##############################################
def download_button(df, filename):
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label=f"📥 Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

##############################################
# DATA LOADERS (UPLOAD + GITHUB)
##############################################
@st.cache_data
def load_default_data():
    # TODO: Replace with your real GitHub raw CSV URL
    url = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/dataset.csv"
    try:
        return pd.read_csv(url)
    except:
        return pd.DataFrame()

st.sidebar.header("📁 Dataset Loader")

uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_default_data()

if df.empty:
    st.error("❌ No dataset available. Upload a CSV file or configure the GitHub URL in load_default_data().")
    st.stop()

##############################################
# NAVIGATION
##############################################
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Data Overview",
        "🤖 Classification",
        "🌀 Clustering",
        "🔗 Association Rule Mining",
        "📈 Regression",
        "💰 Dynamic Pricing",
        "🧬 Persona Generator",
        "📌 Insights"
    ]
)

##############################################
# HOME PAGE
##############################################
if page == "🏠 Home":
    st.markdown("""
        <div style='text-align:center; padding:40px;
            background:linear-gradient(90deg,#A020F0,#00C9FF);
            border-radius:15px; color:white;'>
            <h1 style='font-size:50px;'>AI–VR Fashion Intelligence Suite</h1>
            <h3>The Ultimate All-in-One ML + Pricing + Persona Dashboard</h3>
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🚀 What This Dashboard Includes")
    st.markdown("""
    - ✔ Classification (LR, RF, XGBoost)  
    - ✔ Clustering (KMeans, Hierarchical)  
    - ✔ Association Rule Mining (Apriori)  
    - ✔ Regression (LR, RF, GBR, XGBRegressor)  
    - ✔ Dynamic Pricing Simulator  
    - ✔ Persona Generator  
    - ✔ Insights + Strategies  
    - ✔ Data Upload + Download  
    """)

    st.markdown("---")

##############################################
# DATA OVERVIEW
##############################################
if page == "📊 Data Overview":

    st.header("📊 Dataset Overview")

    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.subheader("Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Stats")
    st.write(df.describe(include="all"))

    st.subheader("📥 Download Dataset")
    download_button(df, "dataset_download.csv")

##############################################
# CLASSIFICATION
##############################################
if page == "🤖 Classification":

    st.header("🤖 Classification Models")

    target = st.selectbox("Target Column", df.columns)
    features = st.multiselect("Feature Columns", df.columns, default=list(df.columns))

    if target in features:
        features.remove(target)

    model_choice = st.selectbox("Choose Algorithm", [
        "Logistic Regression",
        "Random Forest Classifier",
        "XGBoost Classifier"
    ])

    test_size = st.slider("Test Split %", 0.1, 0.5, 0.2)

    if st.button("Train Model"):

        X = pd.get_dummies(df[features])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=500)
        elif model_choice == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            model = XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42
            )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.subheader("📊 Classification Metrics")
        st.write("Accuracy:", accuracy_score(y_test, preds))
        st.write("Precision:", precision_score(y_test, preds, average="weighted", zero_division=0))
        st.write("Recall:", recall_score(y_test, preds, average="weighted", zero_division=0))
        st.write("F1 Score:", f1_score(y_test, preds, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, preds)
        st.subheader("Confusion Matrix")
        st.write(cm)

        out = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        download_button(out, "classification_predictions.csv")

##############################################
# CLUSTERING
##############################################
if page == "🌀 Clustering":

    st.header("🌀 Clustering Models")

    num_clusters = st.slider("Number of Clusters (KMeans)", 2, 10, 3)
    method = st.selectbox("Clustering Method", ["KMeans", "Agglomerative"])

    clustering_features = st.multiselect(
        "Select features for clustering",
        df.columns,
        default=list(df.select_dtypes(include=[np.number]).columns)
    )

    if st.button("Run Clustering"):

        X = df[clustering_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == "KMeans":
            model = KMeans(n_clusters=num_clusters, random_state=42)
        else:
            model = AgglomerativeClustering(n_clusters=num_clusters)

        df["Cluster"] = model.fit_predict(X_scaled)

        st.subheader("Cluster Assigned Data")
        st.dataframe(df.head())

        # PCA 2D Visualization
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        df["PC1"], df["PC2"] = comps[:, 0], comps[:, 1]

        st.subheader("2D Cluster Visualization")
        fig = px.scatter(df, x="PC1", y="PC2", color="Cluster", hover_data=clustering_features)
        st.plotly_chart(fig, use_container_width=True)

        download_button(df, "clustered_data.csv")

##############################################
# ASSOCIATION RULE MINING
##############################################
if page == "🔗 Association Rule Mining":

    st.header("🔗 Association Rule Mining — Apriori Algorithm")

    categorical_cols = st.multiselect(
        "Select Categorical Columns",
        df.columns,
        default=[col for col in df.columns if df[col].dtype == "object"]
    )

    min_support = st.slider("Min Support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5)
    min_lift = st.slider("Min Lift", 0.5, 5.0, 1.0)

    if st.button("Generate Rules"):

        if not categorical_cols:
            st.error("Please select at least one categorical column.")
        else:
            df_hot = pd.get_dummies(df[categorical_cols])

            freq_items = apriori(df_hot, min_support=min_support, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift]

            st.subheader("Generated Rules")
            if rules.empty:
                st.warning("No rules found with the selected thresholds.")
            else:
                st.dataframe(rules)

                download_button(rules, "association_rules.csv")

##############################################
# REGRESSION
##############################################
if page == "📈 Regression":

    st.header("📈 Regression Models")

    target = st.selectbox("Target (Y)", df.columns)
    reg_features = st.multiselect("Features (X)", df.columns, default=list(df.columns))

    if target in reg_features:
        reg_features.remove(target)

    reg_model_choice = st.selectbox(
        "Select Regression Algorithm",
        ["Linear Regression", "Random Forest Regressor",
         "Gradient Boosting Regressor", "XGBoost Regressor"]
    )

    test_size = st.slider("Test Size (%)", 0.1, 0.5, 0.2)

    if st.button("Run Regression"):

        X = pd.get_dummies(df[reg_features])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if reg_model_choice == "Linear Regression":
            model = LinearRegression()
        elif reg_model_choice == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=300, random_state=42)
        elif reg_model_choice == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(random_state=42)
        else:
            model = XGBRegressor(objective="reg:squarederror", random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.subheader("Regression Metrics")
        st.write("R²:", r2_score(y_test, preds))
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
        st.write("MAE:", mean_absolute_error(y_test, preds))

        fig = px.scatter(
            x=y_test, y=preds,
            labels={"x": "Actual", "y": "Predicted"},
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

        out = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        download_button(out, "regression_predictions.csv")

##############################################
# DYNAMIC PRICING
##############################################
if page == "💰 Dynamic Pricing":

    st.header("💰 Dynamic Pricing Simulator")

    cost = st.slider("Product Cost ($)", 10, 300, 60)
    base_margin = st.slider("Base Margin (%)", 5, 80, 30)
    persona_factor = st.slider("Persona Multiplier (%)", 80, 150, 100)
    features_addon = st.slider("AI Feature Add-on ($)", 0, 200, 50)

    dynamic_price = cost * (1 + base_margin / 100) * (persona_factor / 100) + features_addon

    st.success(f"💲 **Recommended Dynamic Price: ${dynamic_price:.2f}**")

    units = np.arange(10, 201, 10)
    revenue = units * dynamic_price

    fig = px.line(x=units, y=revenue, labels={"x": "Units Sold", "y": "Revenue ($)"})
    st.plotly_chart(fig, use_container_width=True)

##############################################
# PERSONA GENERATOR
##############################################
if page == "🧬 Persona Generator":

    st.header("🧬 AI–VR Persona Generator")

    fit_score = st.slider("Fit Importance (1–10)", 1, 10, 5)
    tech_score = st.slider("Tech Comfort (1–10)", 1, 10, 5)
    budget = st.slider("Budget Level (1–5)", 1, 5, 3)
    trendiness = st.slider("Trend Affinity (1–10)", 1, 10, 5)

    if fit_score >= 8 and tech_score >= 7:
        persona = "Premium Perfectionist"
    elif fit_score >= 7 and budget <= 3:
        persona = "Fit Frustrated"
    elif tech_score >= 8 and trendiness >= 7:
        persona = "Metaverse Native"
    elif budget == 1:
        persona = "Budget Conscious"
    else:
        persona = "Eco Warrior"

    st.success(f"🎭 **Your Predicted Persona: {persona}**")

##############################################
# INSIGHTS PAGE
##############################################
if page == "📌 Insights":

    st.header("📌 AI-Generated Insights & Strategy Recommendations")

    st.info("🔥 Target high WTP personas with premium AI–VR bundles.")
    st.warning("⚠ Budget Conscious personas show weaker interest — avoid early targeting.")
    st.success("✨ Metaverse Natives are ideal early adopters for immersive VR try-on.")

    st.subheader("📌 Recommended Go-To-Market Strategy")
    st.markdown("""
    - Focus on **Premium Perfectionists** & **Metaverse Natives**  
    - Lead with **VR Try-On + AI Fit Suggestions**  
    - Build **sustainability and zero-waste stories** for Eco Warriors  
    - Avoid heavy marketing spend on Budget Conscious personas initially  
    - Offer tiered pricing with dynamic add-ons for advanced features  
    """)

##############################################
# FOOTER
##############################################
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>AI–VR Fashion ML Suite • Built with Streamlit</p>",
    unsafe_allow_html=True
)
