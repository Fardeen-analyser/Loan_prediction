import streamlit as st
import pickle
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="SmartLoan AI",
    page_icon="🏦",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

.main-title{
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:700;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:20px;
}

.metric-card{
    background:rgba(255,255,255,0.05);
    padding:20px;
    border-radius:12px;
    text-align:center;
    border-left:4px solid #3b82f6;
}

.result-success{
    background:#14532d;
    padding:20px;
    border-radius:12px;
    color:white;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.result-danger{
    background:#7f1d1d;
    padding:20px;
    border-radius:12px;
    color:white;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Model Loading Failed: {e}")
        return None

model = load_model()

# ======================================================
# HEADER
# ======================================================
st.markdown(
    '<p class="main-title">🏦 SmartLoan AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">AI Powered Loan Eligibility Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ======================================================
# SIDEBAR INPUTS
# ======================================================
st.sidebar.header("Applicant Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.sidebar.selectbox(
    "Marital Status",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    [0, 1, 2, 3]
)

self_employed = st.sidebar.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0.0,
    value=150.0
)

loan_term = st.sidebar.number_input(
    "Loan Amount Term (Months)",
    min_value=0.0,
    value=360.0
)

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

total_income = st.sidebar.number_input(
    "Total Income",
    min_value=0.0,
    value=5000.0
)

# ======================================================
# ENCODING
# ======================================================
gender_val = 1 if gender == "Male" else 0
married_val = 1 if married == "Yes" else 0
self_emp_val = 1 if self_employed == "Yes" else 0

property_val = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}[property_area]

# ======================================================
# DASHBOARD CARDS
# ======================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
        <h3>💰 Income</h3>
        <h2>{total_income:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
        <h3>🏦 Loan Amount</h3>
        <h2>{loan_amount:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
        <h3>📅 Loan Term</h3>
        <h2>{loan_term:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")
st.write("")

# ======================================================
# PREDICTION
# ======================================================
if st.button("🔍 Check Loan Eligibility", use_container_width=True):

    if model is None:
        st.error("Model not available.")
    else:

        input_data = np.array([[
            gender_val,
            married_val,
            dependents,
            self_emp_val,
            loan_amount,
            loan_term,
            property_val,
            total_income
        ]])

        try:
            prediction = model.predict(input_data)

            st.markdown("## Prediction Result")

            if prediction[0] == 1:
                st.markdown(
                    """
                    <div class="result-success">
                    ✅ LOAN APPROVED
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="result-danger">
                    ❌ LOAN REJECTED
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Probability Score
            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(input_data)

                approval_score = probability[0][1] * 100

                st.write("")
                st.subheader("Approval Probability")

                st.progress(float(approval_score) / 100)

                st.info(
                    f"Estimated Approval Chance: **{approval_score:.2f}%**"
                )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")

st.caption(
    "Disclaimer: This prediction is generated by a machine learning model and should be used for educational and decision-support purposes only. Final loan approval is subject to the lender's policies and verification process."
)
