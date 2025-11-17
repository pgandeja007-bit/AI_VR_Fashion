##############################################
#        Meta Fashion – AI–VR ML Dashboard
#                FINAL app.py
##############################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
# PAGE CONFIG
##############################################
st.set_page_config(
    page_title="Meta Fashion – AI–VR Fashion Intelligence Suite",
    layout="wide"
)

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
# DATA LOADERS (UPLOAD + LOCAL DEFAULT CSV)
##############################################
@st.cache_data
def load_default_data():
    """
    Loads the default AI–VR Fashion Survey dataset.
    Make sure the file:
    'AI_VR_Fashion_Survey_Synthetic_Data (1).csv'
    is in the same folder as this app.py.
    """
    return pd.read_csv("AI_VR_Fashion_Survey_Synthetic_Data (1).csv")

st.sidebar.header("📁 Dataset Loader")

uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = load_default_data()

if df.empty:
    st.error("❌ No dataset available. Upload a CSV file or ensure the default CSV is present.")
    st.stop()

##############################################
# NAVIGATION
##############################################
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🏢 Meta Fashion Info",
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
            <h1 style='font-size:50px;'>Meta Fashion – AI–VR Fashion Intelligence Suite</h1>
            <h3>Data-Driven Insights for an AI–VR Powered Fashion Brand</h3>
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🚀 What This Dashboard Includes")
        st.markdown("""
        - ✔ Classification (Logistic Regression, Random Forest, XGBoost)  
        - ✔ Clustering (KMeans, Agglomerative + PCA)  
        - ✔ Association Rule Mining (Apriori)  
        - ✔ Regression (Linear, RF, GBR, XGBoost Regressor)  
        - ✔ Dynamic Pricing Simulator (scenario analysis)  
        - ✔ Persona Generator (Fit Frustrated, Premium Perfectionist, Metaverse Native, etc.)  
        - ✔ Insights aligned to survey results and business model  
        - ✔ Data Upload + Download  
        """)

    # Company quick info card on Home (fixed colors + Meta Fashion)
    with col2:
        st.markdown("""
        <div style='padding:18px; border-radius:14px; background:#252545; border:1px solid #555; color:#f5f5f5;'>
            <h4 style='color:#ffffff; margin-bottom:8px;'>🏢 Meta Fashion</h4>
            <p style='font-size:14px; line-height:1.4; color:#f5f5f5;'>
            Meta Fashion is an <b>AI–VR fashion platform</b> that turns any outfit idea into a 
            real, perfectly fitted garment through a fully digital pipeline:
            </p>
            <ol style='font-size:13px; color:#f5f5f5; padding-left:18px;'>
                <li><b>AI Design</b> – AI generates 3D outfit concepts and trend-aligned styles.</li>
                <li><b>VR Fitting</b> – Fit, drape and movement are tested on a virtual avatar.</li>
                <li><b>Digital to Physical</b> – 3D garments are converted into accurate 2D patterns.</li>
                <li><b>Real Fit</b> – Final garments match the tuned virtual fit with minimal waste.</li>
            </ol>
            <p style='font-size:13px; color:#f5f5f5;'>
            The promise: <b>Creative Freedom + Perfect Fit + Near Zero-Waste</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

##############################################
# COMPANY INFO PAGE
##############################################
if page == "🏢 Meta Fashion Info":
    st.header("🏢 Meta Fashion – Company Profile")

    st.subheader("1. Business Concept")
    st.markdown("""
    **Meta Fashion** is an AI–VR powered fashion brand that converts a customer's or designer's idea  
    into a **real, perfectly fitted garment** using a fully digital pipeline.
    Instead of relying on multiple physical samples, fit trials, and guesswork, the
    system uses AI and VR to validate designs before a single piece of fabric is cut.
    """)

    st.subheader("2. Four-Step Process")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Step 1 – AI Design**  
        - AI helps create first-draft 3D designs and predicts trends.  
        - Captures creative preferences and style choices.  

        **Step 2 – VR Fitting**  
        - Fit, drape and movement are tested on a virtual avatar in VR.  
        - All corrections are made digitally (no fabric waste).  
        """)
    with col2:
        st.markdown("""
        **Step 3 – Digital to Physical**  
        - The optimised 3D model is converted into an accurate 2D pattern.  
        - This bridges the digital garment and the physical pattern.  

        **Step 4 – Real Fit Production**  
        - The final garment is manufactured to match the digital fit.  
        - The result is a real outfit that mirrors the tuned virtual version.  
        """)

    st.subheader("3. Why This Is Useful")
    st.markdown("""
    Meta Fashion directly addresses three major pain points in fashion:

    - **Unlimited Creativity & Customisation**  
      Fashion-conscious, creative customers can convert their ideas into custom outfits
      with AI support, instead of being limited to standard designs.

    - **Fit Nightmare Solution**  
      Customers who struggle with fit get a better and more reliable fit. This reduces  
      returns for e-commerce and increases satisfaction.

    - **Sustainability & Speed**  
      Multiple virtual fittings replace physical samples, cutting fabric waste and
      shortening the design cycle significantly.
    """)

    st.subheader("4. Unique Selling Proposition (USP)")
    st.markdown("""
    Meta Fashion's USP combines:
    - **Creative Freedom** – customers and designers co-create unique looks.  
    - **Perfect Fit Guarantee** – fit is tested and tuned digitally before production.  
    - **True Digital-First Production** – the 3D model is directly converted into a
      2D pattern, aligning virtual and physical garments closely.
    """)

    st.subheader("5. Target Market & Key Personas")
    st.markdown("""
    Based on the survey and persona analysis, Meta Fashion focuses on three primary personas:

    - **Fit Frustrated (Prime Target)**  
      - High importance of fit, high difficulty in finding the right fit.  
      - Strong interest in a solution that can guarantee fit and reduce pain of returns.  

    - **Premium Perfectionist**  
      - Higher income and willingness to pay a premium.  
      - Values quality, craftsmanship, and exclusivity.  

    - **Metaverse Native**  
      - Tech-comfortable, excited by AI, VR, and immersive experiences.  
      - Strong interest in AI Design Suggestions and VR Try-On.

    Initial focus: **high-price, fit-critical products** such as suits and blazers.
    These categories have the highest willingness to pay and benefit most from perfect fit.
    """)

    st.subheader("6. Value Proposition")
    st.markdown("""
    > **"Bring Your Unique Vision to Life with Guaranteed Precision and Near Zero-Waste."**  

    - For customers: tailor-made garments that match their creative vision and fit requirements.  
    - For the planet: reduced fabric waste and more efficient production.  
    """)

    st.subheader("7. Why This Is a Strong Data Analytics Project")
    st.markdown("""
    The entire Meta Fashion model runs on data-driven decision making:

    - **AI Design Data** – trends, style choices, and creative preferences feed into
      models that refine design suggestions.  

    - **VR Fit Data** – detailed fit and movement feedback is captured digitally and
      used to improve pattern accuracy and reduce returns.  

    - **Customer Behaviour Data** – survey responses about interest, price sensitivity,
      feature preference, and persona type help decide which segments and products to
      prioritise.

    This is why Meta Fashion is an ideal use case for **classification, clustering,
    association rule mining, and regression** to guide strategy, pricing, product
    design, and rollout.
    """)

    st.subheader("8. Role of This Dashboard")
    st.markdown("""
    This Streamlit dashboard acts as a **decision cockpit** for Meta Fashion:

    - Segments users (clustering)  
    - Predicts interest or adoption (classification)  
    - Estimates willingness to pay (regression)  
    - Finds feature affinity patterns (association rules)  
    - Simulates pricing and revenue (dynamic pricing)  
    - Summarises insights into a business narrative (Insights tab)  
    """)

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

        if len(features) == 0:
            st.error("Please select at least one feature column.")
            st.stop()

        X = pd.get_dummies(df[features])
        y = df[target]

        # Drop rows with missing values in X or y
        data = pd.concat([X, y], axis=1).dropna()
        X = data[X.columns]
        y = data[target]

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

        if len(clustering_features) == 0:
            st.error("Please select at least one feature column for clustering.")
            st.stop()

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
# REGRESSION (NUMERIC-ONLY TARGET)
##############################################
if page == "📈 Regression":

    st.header("📈 Regression Models")

    # Only allow numeric columns as valid targets for regression
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.error("No numeric columns available for regression. Please upload a dataset with numeric targets.")
        st.stop()

    target = st.selectbox("Target (Y - numeric only)", numeric_cols)

    # Features can be any columns except the target
    possible_features = [col for col in df.columns if col != target]
    reg_features = st.multiselect(
        "Features (X)",
        possible_features,
        default=possible_features
    )

    reg_model_choice = st.selectbox(
        "Select Regression Algorithm",
        ["Linear Regression", "Random Forest Regressor",
         "Gradient Boosting Regressor", "XGBoost Regressor"]
    )

    test_size = st.slider("Test Size (%)", 0.1, 0.5, 0.2)

    if st.button("Run Regression"):

        if len(reg_features) == 0:
            st.error("Please select at least one feature column.")
            st.stop()

        # Work on subset and drop rows with missing target/feature values
        data = df[reg_features + [target]].dropna()
        if data.empty:
            st.error("No data left after dropping missing values. Check your selected columns.")
            st.stop()

        X = pd.get_dummies(data[reg_features])
        y = data[target].astype(float)  # safe: target is numeric by construction

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Model selection
        if reg_model_choice == "Linear Regression":
            model = LinearRegression()
        elif reg_model_choice == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=300, random_state=42)
        elif reg_model_choice == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(random_state=42)
        else:
            model = XGBRegressor(objective="reg:squarederror", random_state=42)

        # Fit model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        st.subheader("Regression Metrics")
        st.write("R²:", r2_score(y_test, preds))
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
        st.write("MAE:", mean_absolute_error(y_test, preds))

        # Plot Actual vs Predicted
        st.subheader("Actual vs Predicted")
        fig = px.scatter(
            x=y_test,
            y=preds,
            labels={"x": "Actual", "y": "Predicted"},
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download predictions
        out = pd.DataFrame({"Actual": y_test, "Predicted": preds})
        download_button(out, "regression_predictions.csv")

##############################################
# DYNAMIC PRICING
##############################################
if page == "💰 Dynamic Pricing":

    st.header("💰 Dynamic Pricing Simulator – Meta Fashion")

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

    st.header("🧬 Meta Fashion Persona Generator")

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
# INSIGHTS PAGE (ALIGNED TO META FASHION)
##############################################
if page == "📌 Insights":

    st.header("📌 Key Insights for Meta Fashion")

    st.subheader("1. Persona Distribution & Interest")
    st.markdown("""
    - The largest persona groups are **Budget Conscious**, **Premium Perfectionist**, and **Fit Frustrated**.  
    - **Metaverse Natives** and **Eco Warriors** are smaller but highly strategic segments.  
    - Metaverse Natives show the highest share of early adopters, while Fit Frustrated show very high overall interest, 
      strongly supporting Meta Fashion’s focus on solving fit problems with AI–VR.
    """)

    st.subheader("2. Fit Importance vs Difficulty (Prime Target)")
    st.markdown("""
    - Customers who rate fit as **very important and very difficult** form the **Prime Target** segment.  
    - Many of these customers are highly interested in the concept of Meta Fashion.  
    - Customers with low fit difficulty or low fit importance are secondary segments, not immediate focus.
    """)

    st.subheader("3. Feature Preference – What Customers Want")
    st.markdown("""
    - **Custom Fit Guarantee** is the strongest feature across segments, especially for Fit Frustrated,  
      Eco Warriors and Premium Perfectionists.  
    - **AI Design Suggestions** and **VR Try-On** resonate strongly with Metaverse Natives and Premium Perfectionists.  
    - Budget Conscious users rate most features lower, signalling that they need value-focused or basic offerings.
    """)

    st.subheader("4. Willingness to Pay by Persona & Category")
    st.markdown("""
    - Highest willingness to pay appears in **Suits/Blazers** – a natural first category for Meta Fashion.  
    - **Premium Perfectionist** and **Fit Frustrated** are willing to pay the most for these premium categories.  
    - Budget Conscious users lean towards lower price brackets, so they are not the primary target for high-end offerings.
    """)

    st.subheader("5. Customer Journey Insights")
    st.markdown("""
    - A high proportion of respondents report **fit issues**, validating the problem Meta Fashion wants to solve.  
    - A strong share of customers is **tech comfortable**, making AI–VR a viable interface.  
    - A meaningful portion of interested users is willing to pay more and wants **early access**, showing a real early adopter base.
    """)

    st.subheader("6. Strategic Recommendations for Meta Fashion")
    st.markdown("""
    - **Primary Target Segments:**  
      - Fit Frustrated (pain-driven, fit-first)  
      - Premium Perfectionist (quality & premium experience)  
      - Metaverse Native (tech enthusiasm and viral potential)  

    - **Product Focus:**  
      - Launch Meta Fashion with **Suits/Blazers** and other fit-critical categories.  
      - Emphasise **Custom Fit Guarantee + AI Design + VR Try-On** as the core bundle.

    - **Positioning & Pricing:**  
      - Justify higher prices by emphasising perfect fit, unique designs, and sustainability.  
      - Use dynamic pricing experiments to find a sweet spot for each persona.

    - **Risk/Watch-Out:**  
      - Budget Conscious segment is highly price sensitive; consider a simpler, more affordable path for them later.
    """)

##############################################
# FOOTER
##############################################
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Meta Fashion • AI–VR Fashion Intelligence Suite • Built with Streamlit</p>",
    unsafe_allow_html=True
)
