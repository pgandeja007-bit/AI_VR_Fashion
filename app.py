# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI–VR Fashion Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("AI_VR_Fashion_Survey_Synthetic_Data (1).csv")
    return df

df = load_data()

# Map canonical interest strings to an ordinal score (safe handling)
interest_map = {
    "Not interested - I prefer traditional shopping": 1,
    "Slightly interested - I might try it eventually": 2,
    "Moderately interested - I'd consider it": 3,
    "Very interested - I'd definitely try it": 4,
    "Extremely interested - I'd be an early adopter": 5
}
df["InterestScore"] = df["Q24_InterestLevel"].map(interest_map)

st.sidebar.title("AI–VR Fashion — Dashboard")
st.sidebar.caption("Interactive analytics for the AI+VR fashion concept")
view = st.sidebar.radio(
    "Choose view",
    [
        "Overview & KPIs",
        "Customer Personas & Interest",
        "Fit Importance vs Difficulty",
        "Purchase Factors vs Interest",
        "Feature Interest Heatmap",
        "Willingness to Pay (Suits / Blazers)",
        "Customer Journey Funnel",
        "Executive Summary"
    ],
    index=0
)

if view == "Overview & KPIs":
    st.title("Overview & Key Metrics")
    st.write("""
    Summary of a 600-respondent survey on an AI+VR fashion platform (virtual try-on, AI design suggestions,
    custom fit guarantee).
    """)
    total = len(df)
    early_adopters = int((df["Q24_InterestLevel"] == "Extremely interested - I'd be an early adopter").sum())
    tech_positive = int(df["Q15_TechComfort"].isin([
        "Comfortable - I adopt new tech fairly quickly",
        "Early adopter - I'm usually first to try new tech"
    ]).sum())
    prime_target = int(((df["Q12_FitImportance"] >= 7) & (df["Q13_FitDifficulty"] >= 6)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total respondents", f"{total}")
    c2.metric("Early adopters", f"{early_adopters}", f"{early_adopters/total:.0%}")
    c3.metric("Tech-comfortable users", f"{tech_positive}", f"{tech_positive/total:.0%}")
    c4.metric("Prime target (fit-critical)", f"{prime_target}", f"{prime_target/total:.0%}")

    st.subheader("Persona distribution")
    persona_counts = df["Persona"].value_counts().reset_index()
    persona_counts.columns = ["Persona", "Count"]
    fig = px.pie(persona_counts, names="Persona", values="Count", hole=0.35)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interest level distribution")
    fig2 = px.histogram(df, x="Q24_InterestLevel",
                        category_orders={"Q24_InterestLevel": list(interest_map.keys())})
    fig2.update_layout(xaxis_title="Interest level", yaxis_title="Respondents")
    st.plotly_chart(fig2, use_container_width=True)

elif view == "Customer Personas & Interest":
    st.title("Customer Personas & Interest")
    st.write("Stacked view of interest levels by persona (who are the likely early adopters).")
    fig = px.histogram(
        df,
        x="Persona",
        color="Q24_InterestLevel",
        barmode="stack",
        category_orders={"Q24_InterestLevel": list(interest_map.keys())}
    )
    fig.update_layout(xaxis_title="Persona", yaxis_title="Respondents", legend_title="Interest")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insights**
    - Metaverse Natives & Premium Perfectionists: highest 'Very'/'Extremely interested' shares.
    - Fit Frustrated: strong interest — good practical target.
    - Budget Conscious: more price-sensitive (lower extreme-interest share).
    """)

elif view == "Fit Importance vs Difficulty":
    st.title("Fit Importance vs Fit Difficulty – Quadrant")
    st.write("Top-right quadrant = high importance and high difficulty (prime target).")
    fig = px.scatter(
        df, x="Q12_FitImportance", y="Q13_FitDifficulty",
        color="Q24_InterestLevel",
        hover_data=["Persona", "Q8_AnnualSpend"],
        category_orders={"Q24_InterestLevel": list(interest_map.keys())}
    )
    fig.add_shape(type="line", x0=7, x1=7, y0=df["Q13_FitDifficulty"].min(), y1=df["Q13_FitDifficulty"].max(),
                  line=dict(dash="dash"))
    fig.add_shape(type="line", x0=df["Q12_FitImportance"].min(), x1=df["Q12_FitImportance"].max(), y0=6, y1=6,
                  line=dict(dash="dash"))
    fig.update_layout(xaxis_title="Fit importance (1–10)", yaxis_title="Fit difficulty (1–10)")
    st.plotly_chart(fig, use_container_width=True)
    prime_share = ((df["Q12_FitImportance"] >= 7) & (df["Q13_FitDifficulty"] >= 6)).mean()
    st.write(f"**About {prime_share:.0%}** of respondents are in the prime target quadrant.")

elif view == "Purchase Factors vs Interest":
    st.title("Purchase Factors vs Interest (correlation)")
    st.write("Correlation between Q17_* purchase factors and InterestScore (1–5). Positive = higher interest.")
    factor_cols = [c for c in df.columns if c.startswith("Q17_")]
    corr_list = []
    for c in factor_cols:
        try:
            corr = df["InterestScore"].corr(df[c])
            corr_list.append({"factor": c, "corr": float(corr) if pd.notna(corr) else 0.0})
        except Exception:
            corr_list.append({"factor": c, "corr": 0.0})
    corr_df = pd.DataFrame(corr_list).sort_values("corr", ascending=True)
    label_map = {
        "Q17_Price_Value": "Price / Value",
        "Q17_Perfectfit": "Perfect Fit",
        "Q17_Brandreputation": "Brand Reputation",
        "Q17_Sustainability_ecofriendliness": "Sustainability / Eco-friendliness",
        "Q17_Unique_customdesigns": "Unique Custom Designs",
        "Q17_Latesttrends": "Latest Trends",
        "Q17_Comfort": "Comfort",
        "Q17_Quality_durability": "Quality / Durability",
        "Q17_Fastdelivery": "Fast Delivery",
        "Q17_Easyreturns": "Easy Returns",
        "Q17_Ethicalmanufacturing": "Ethical Manufacturing",
        "Q17_Celebrity_influencerendorsements": "Celebrity / Influencer",
        "Q17_Versatility_multiuse": "Versatility / Multi-use",
        "Q17_Madetoorder_personalization": "Made-to-order Personalization"
    }
    corr_df["label"] = corr_df["factor"].map(label_map).fillna(corr_df["factor"])
    fig = px.bar(corr_df, x="corr", y="label", orientation="h")
    fig.update_layout(xaxis_title="Correlation with interest", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    - Perfect fit & personalization: strongest positive drivers.
    - Price / value: strongest negative driver — pricing must be tested.
    """)

elif view == "Feature Interest Heatmap":
    st.title("Feature Interest Heatmap by Persona")
    st.write("Average persona ratings (1–5) for key platform features.")
    feature_cols = [
        "Q23_AIgenerateddesignsuggestions",
        "Q23_VirtualtryoninVR",
        "Q23_Customfitguarantee",
        "Q23_Zerowasteproduction",
        "Q23_Digitalwardrobemanagement",
        "Q23_DesignyourownclotheswithAI"
    ]
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        st.error("No matching feature columns found in the dataset.")
    else:
        heat = df.groupby("Persona")[existing].mean().T
        rename = {
            "Q23_AIgenerateddesignsuggestions": "AI design suggestions",
            "Q23_VirtualtryoninVR": "VR try-on",
            "Q23_Customfitguarantee": "Custom fit guarantee",
            "Q23_Zerowasteproduction": "Zero-waste production",
            "Q23_Digitalwardrobemanagement": "Digital wardrobe",
            "Q23_DesignyourownclotheswithAI": "Design with AI"
        }
        heat = heat.rename(columns=rename)
        fig = px.imshow(heat, labels=dict(x="Persona", y="Feature", color="Average rating"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Metaverse Natives and Premium Perfectionists show relatively higher interest in advanced features.")

elif view == "Willingness to Pay (Suits / Blazers)":
    st.title("Willingness to Pay — Suits / Blazers")
    st.write("Distribution of reported willingness to pay by persona (violin plot).")
    if "Q27_SuitPrice" not in df.columns:
        st.error("Q27_SuitPrice column not found in dataset.")
    else:
        fig = px.violin(df, x="Persona", y="Q27_SuitPrice", box=True, points="all")
        fig.update_layout(xaxis_title="Persona", yaxis_title="Willingness to pay (suit/blazer)")
        st.plotly_chart(fig, use_container_width=True)

elif view == "Customer Journey Funnel":
    st.title("Customer Journey Funnel")
    st.write("Rough funnel: Awareness → Interested → Beta sign-ups → Willing to pay.")
    aware = len(df)
    interested = int((df["InterestScore"] >= 3).sum())
    beta = int((df["Q37_BetaTestInterest"] == "Yes").sum()) if "Q37_BetaTestInterest" in df.columns else int((df["InterestScore"] >= 4).sum() * 0.25)
    willing_pay = int((df["InterestScore"] >= 4).sum())
    funnel_df = pd.DataFrame({
        "stage": ["Awareness", "Interested (3+)", "Beta sign-ups (est)", "Willing to pay (4+)"],
        "count": [aware, interested, beta, willing_pay]
    })
    fig = px.funnel(funnel_df, x="count", y="stage")
    st.plotly_chart(fig, use_container_width=True)

elif view == "Executive Summary":
    st.title("Executive summary & recommended next steps")
    st.markdown("""
    **Top insights**
    - Strong interest: large majority are moderately to extremely interested.
    - Best early segments: Metaverse Natives, Premium Perfectionists, Fit Frustrated.
    - Barrier: Price/value concerns — pricing & packaging need tests.
    - Feature priorities: Perfect fit, personalization, VR try-on, AI design suggestions.

    **Recommended next steps**
    1. Run a paid beta focusing on suits/blazers for Premium and Fit Frustrated segments.
    2. A/B test pricing and include a Fit Guarantee to reduce friction.
    3. Collect digital→physical fit delta metrics to quantify accuracy.
    4. Use classification to prioritize outreach to high-conversion respondents.
    """)
    st.markdown("Deploy: push to GitHub and deploy on Streamlit Cloud (or Heroku).")

else:
    st.write("Select a view from the sidebar.")
