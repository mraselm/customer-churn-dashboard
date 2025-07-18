import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stExpander {
          background-color: #1e1e1e !important;
          border: 1px solid #444 !important;
          border-radius: 6px !important;
          padding: 0.5rem;
        }
        .stExpanderHeader {
          font-weight: 600;
          color: #f0f0f0;
        }
        .stExpander + .stExpander {
          margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        width: 260px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
  <style>
  .css-1v0mbdj.e1tzin5v2 {
      width: 50px !important;
      min-width: 50px !important;
  }
  </style>
""", unsafe_allow_html=True)

# --- Load model and data ---
model = xgb.Booster()
model.load_model("best_xgboost_model.json")

if "data" not in st.session_state:
    df = pd.read_csv("processed_churn_dataset.csv")
    st.session_state.data = df
else:
    df = st.session_state.data

X = df.drop("Churn", axis=1)
y = df["Churn"]

# --- Header ---
st.title("Customer Churn Prediction Dashboard")
st.markdown("Get detailed insights into customer churn predictions using ML and SHAP explanations.")

col1, col2 = st.columns(2)
left_col, middle_col, right_col = st.columns([2, 2.5, 1.5])

with col2:
    st.sidebar.title("🧭 Select or Upload")
    st.sidebar.subheader("📌 Customer Selection")
    customer_ids = X.index.tolist()
    selected_id = st.sidebar.selectbox("Choose Customer ID", customer_ids)

    st.sidebar.subheader("⚙️ Prediction Settings")
    threshold = st.sidebar.slider("Churn threshold", 0.0, 1.0, 0.5)

    # Show threshold guidance text
    if threshold <= 0.5:
        st.sidebar.markdown("""
        **Goal**: Minimize churn at all costs  
        **Threshold**: 0.3 - 0.5  
        **Insight**: Flag more customers to retain borderline cases.
        """)
    elif threshold <= 0.6:
        st.sidebar.markdown("""
        **Goal**: Balance retention effort and cost  
        **Threshold**: 0.5 - 0.6  
        **Insight**: Flag moderately confident churners only.
        """)
    else:
        st.sidebar.markdown("""
        **Goal**: Focus only on high-risk churners  
        **Threshold**: 0.7 - 0.9  
        **Insight**: Save effort by targeting top churn risks only.
        """)

    uploaded_file = st.sidebar.file_uploader("📦 Upload new customer file", type=["csv"])
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        st.sidebar.write("Preview:")
        st.sidebar.dataframe(new_df.head())
        st.session_state.data = new_df
        st.experimental_rerun()

customer_data = X.iloc[selected_id:selected_id+1]  # DataFrame with one row

# --- Model Prediction ---
dtest = xgb.DMatrix(customer_data, feature_names=customer_data.columns.tolist())
prediction_proba = model.predict(dtest)[0]
prediction_label = "🔴 Likely to Churn" if prediction_proba > threshold else "🟢 Likely to Stay"
confidence_level = "High" if abs(prediction_proba - threshold) > 0.25 else "Medium" if abs(prediction_proba - threshold) > 0.1 else "Low"

# --- Layout Grid ---

with left_col:
    st.subheader("🎯 Prediction Result")
    st.metric(label="Prediction", value=prediction_label)
    st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
    st.metric(label="Model Confidence", value=confidence_level)

with middle_col:
    with st.expander("📄 Customer Profile", expanded=True):
        st.subheader("📄 Customer Profile")
        profile_display = customer_data.T
        profile_display.columns = ["Value"]
        st.dataframe(profile_display, height=150, use_container_width=True)

    with st.expander("💡 Suggested Action & Insights", expanded=True):
        st.subheader("💡 Suggested Action")
        st.markdown(" ")

        # Risk tier
        if prediction_proba > 0.7:
            risk_level = "High"
        elif prediction_proba > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Get top SHAP features and values (up to 3)
        explainer = shap.Explainer(model)
        shap_values = explainer(customer_data)
        top_indices = np.argsort(np.abs(shap_values[0].values))[::-1][:3]
        top_features = [shap_values[0].feature_names[i] for i in top_indices]
        top_values = [shap_values[0].data[i] for i in top_indices]

        # Define tailored insight per feature based on actual value
        feature_actions = {
            "CashbackAmount": lambda v: "Low cashback last month. Offer bonus rewards or limited-time deals." if v < 100 else "Engaged cashback user. Continue incentives or loyalty perks.",
            "Tenure": lambda v: "New customer. Send onboarding email or welcome kit." if v < 6 else "Loyal customer. Offer exclusive VIP benefits or loyalty rewards.",
            "Complain": lambda v: "Complaint raised last month. Send apology and discount voucher." if v == 1 else "No recent complaint. Keep experience consistent.",
            "CouponUsed": lambda v: "No coupons used recently. Push discount codes or time-limited offers." if v == 0 else "Active coupon user. Keep sending personalized offers.",
            "OrderCount": lambda v: "Few orders last month. Suggest bundles or increase engagement through reminders." if v < 3 else "Healthy order frequency. Maintain product suggestions.",
            "NumberOfAddress": lambda v: "Multiple shipping addresses may indicate address instability. Validate preferences or offer flexible delivery options.",
            "DaySinceLastOrder": lambda v: "Customer inactive for a while. Send reactivation offer or best-selling product reminder." if v > 30 else "Customer recently ordered. Follow up with feedback request.",
            "EngagementScore": lambda v: "Low engagement score. Recommend re-engagement content via newsletter." if v < 50 else "High engagement. Suggest new product line or loyalty bonus.",
            "SatisfactionScore": lambda v: "Low satisfaction score. Trigger follow-up survey or apology offer." if v < 3 else "Satisfied customer. Encourage reviews or referrals.",
            "HourSpendOnApp": lambda v: "Low app usage. Push new feature highlights or UX improvements." if v < 1 else "Active app user. Promote premium tools or referral options."
        }

        insights = []
        for f, v in zip(top_features, top_values):
            func = feature_actions.get(f, lambda v: "General churn risk. Re-engage wisely.")
            insights.append(f"• **{f}** — {func(v)}")

        if prediction_proba > 0.7:
            st.warning(f"This customer is at **high risk** of churning.\n\nTop factors:\n- **{top_features[0]}**\n- **{top_features[1]}**\n\n🎯 Suggest: Retention campaign, personalized outreach.")
        elif prediction_proba > 0.3:
            st.info(f"Moderate churn risk.\n\nTop driver: **{top_features[0]}**\n\n📩 Suggest: Gentle email nudge or loyalty offer.")
        else:
            st.success(f"Low churn risk.\n\nTop driver: **{top_features[0]}** (not concerning).\n\n✅ Maintain current engagement.")

        st.markdown(" ")
        st.info("🧠 Business Insight:\n" + "\n".join(insights))

with right_col:
    with st.expander("🧠 Explanation & SHAP Summary", expanded=True):
        # --- SHAP Explanation ---
        st.subheader("🧠 Why This Prediction?")
        explainer = shap.Explainer(model)
        shap_values = explainer(customer_data)

        fig, ax = plt.subplots(figsize=(5, 2.2))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig, use_container_width=True)

        # --- SHAP Global Summary ---
        st.subheader("🌐 SHAP Global Summary (Top 10 Features)")
        sample_X = X.sample(200, random_state=42)
        sample_dmatrix = xgb.DMatrix(sample_X, feature_names=X.columns.tolist())
        explainer = shap.Explainer(model)
        global_shap_values = explainer(sample_X)

        fig2, ax2 = plt.subplots(figsize=(5, 2.2))
        shap.plots.bar(global_shap_values, max_display=10, show=False)
        st.pyplot(fig2, use_container_width=True)

st.markdown("---")
st.caption("©️️ Built by Rasel Mia | Powered by XGBoost, SHAP, and Streamlit")