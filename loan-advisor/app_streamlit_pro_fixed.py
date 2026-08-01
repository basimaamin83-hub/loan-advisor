import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64
import os
import traceback

# ========================
# إعدادات الصفحة المتقدمة
# ========================
st.set_page_config(
    page_title="Smart Loan Advisor Pro",
    page_icon="\U0001F3DB",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================
# CSS المخصص للتصميم الاحترافي
# ========================
st.markdown("""
<style>
    /* خلفية الصفحة الرئيسية مع تأثير الزجاج */
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 0px !important;
    }
    
    /* شريط العنوان العلوي */
    .header-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px 40px;
        margin: -60px -50px 20px -50px;
        border-radius: 0px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* البطاقات الزجاجية */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 215, 0, 0.3);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* عنوان القسم */
    .section-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(255, 215, 0, 0.3);
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* بطاقات المؤشرات */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        border-color: rgba(255, 215, 0, 0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* تنسيق المدخلات */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 8px 15px !important;
    }
    
    /* زر التوقع */
    .predict-btn {
        background: linear-gradient(135deg, #f7971e, #ffd200) !important;
        color: #000 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 12px 40px !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 8px 25px rgba(247, 151, 30, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .predict-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(247, 151, 30, 0.6) !important;
    }
    
    /* صندوق النتيجة */
    .result-box {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-top: 20px;
    }
    
    /* شريط التقدم */
    .progress-bar {
        height: 8px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #00b894, #fdcb6e, #e17055);
        transition: width 1s ease;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# خلفية بسيطة متوافقة مع Streamlit
# ========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    h1, h2, h3, p, label, span {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# تحميل النموذج مع تحقُّق من المسارات والتعامل مع الأخطاء
# ========================
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'loan_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    features_path = os.path.join('models', 'features.pkl')
    le_path = os.path.join('models', 'label_encoders.pkl')

    print("DEBUG: Running load_model(). cwd=", os.getcwd())
    print("DEBUG: Checking files in models/ ->", os.listdir('models') if os.path.exists('models') else 'models folder missing')

    missing = []
    for p in [model_path, scaler_path, features_path, le_path]:
        exists = os.path.exists(p)
        print(f"DEBUG: exists {p}: {exists}")
        if not exists:
            missing.append(p)

    if missing:
        err = f"Missing model files: {missing}"
        print("ERROR:", err)
        return None, None, None, None

    try:
        model = joblib.load(model_path)
        print("DEBUG: model loaded type:", type(model))
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        label_encoders = joblib.load(le_path)
        print("DEBUG: scaler, features, label_encoders loaded")
        return model, scaler, features, label_encoders
    except Exception as e:
        print("ERROR: Exception while loading models:", e)
        traceback.print_exc()
        return None, None, None, None

model, scaler, features, label_encoders = load_model()
if model is None:
    st.error("خطأ: فشل تحميل ملفات النماذج. راجع الطرفية (terminal) لمزيد من التفاصيل.")

# ========================
# الهيدر
# ========================
st.title("🏛️ Smart Loan Advisor")
st.caption("AI-Powered Credit Risk Assessment Platform")
st.divider()

# ========================
# تخطيط الصفحة الرئيسية
# ========================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Loan Application")
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        loan_amnt = st.number_input("💰 Loan Amount ($)", min_value=1000, max_value=100000, value=20000, step=1000)
    with row1_col2:
        annual_inc = st.number_input("💵 Annual Income ($)", min_value=10000, max_value=500000, value=60000, step=1000)
    
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        int_rate = st.slider("📈 Interest Rate (%)", 5.0, 25.0, 12.0, 0.5)
    with row2_col2:
        dti = st.slider("📊 Debt-to-Income (%)", 0.0, 50.0, 15.0, 0.5)
    
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        fico_score = st.slider("📋 FICO Score", 580, 850, 700, 5)
    with row3_col2:
        emp_length = st.slider("💼 Work Experience (Years)", 0, 30, 5, 1)
    
    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        purpose = st.selectbox("🎯 Purpose", [
            "debt_consolidation", "home_improvement", "major_purchase",
            "medical", "car", "credit_card", "small_business", "other"
        ])
    with row4_col2:
        grade = st.selectbox("⭐ Grade", ["A", "B", "C", "D", "E", "F", "G"])
    
    home_ownership = st.selectbox("🏠 Home Ownership", ["RENT", "OWN", "MORTGAGE"])

with col_right:
    st.subheader("📊 Financial Details")
    
    monthly_payment = loan_amnt * (int_rate/100/12) * (1 + int_rate/100/12)**36 / ((1 + int_rate/100/12)**36 - 1)
    
    m1, m2 = st.columns(2)
    m1.metric("Monthly Payment", f"${monthly_payment:,.0f}")
    m2.metric("Payment-to-Income", f"{monthly_payment*12/annual_inc*100:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        revol_bal = st.number_input("🔄 Revolving Balance ($)", min_value=0, max_value=100000, value=10000, step=1000)
        revol_util = st.slider("📊 Revolving Utilization (%)", 0.0, 100.0, 30.0, 1.0)
    with col2:
        open_acc = st.number_input("📂 Open Accounts", min_value=0, max_value=50, value=5, step=1)
        total_acc = st.number_input("📁 Total Accounts", min_value=0, max_value=100, value=10, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        delinq_2yrs = st.number_input("⚠️ Delinquencies (2yrs)", min_value=0, max_value=10, value=0, step=1)
    with col2:
        pub_rec = st.number_input("📜 Public Records", min_value=0, max_value=5, value=0, step=1)

if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
    try:
        grade_encoded = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}[grade]
        purpose_encoded = {
            "debt_consolidation": 0, "home_improvement": 1, "major_purchase": 2,
            "medical": 3, "car": 4, "credit_card": 5, "small_business": 6, "other": 7
        }[purpose]
        home_ownership_encoded = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}[home_ownership]
        term_encoded = 0

        if model is None or scaler is None or features is None:
            # graceful fallback: compute simple heuristic so UI shows results
            print("DEBUG: model/scaler/features missing; using heuristic fallback")
            if fico_score > 700 and dti < 20 and int_rate < 15:
                proba = 0.12
            elif fico_score > 650:
                proba = 0.35
            else:
                proba = 0.72
            prediction = "Bad" if proba > 0.5 else "Good"
        else:
            input_data = pd.DataFrame([[
                loan_amnt, int_rate, monthly_payment, dti, delinq_2yrs,
                open_acc, pub_rec, revol_bal, revol_util, total_acc,
                fico_score, emp_length,
                term_encoded, grade_encoded, home_ownership_encoded,
                0, purpose_encoded
            ]], columns=features)
            input_scaled = scaler.transform(input_data)
            proba = model.predict_proba(input_scaled)[0][1]
            prediction = "Bad" if proba > 0.5 else "Good"
    except Exception as e:
        st.error("خطأ أثناء التنبؤ: " + str(e))
        st.text(traceback.format_exc())
        proba = 0.5
        prediction = "Bad"
    
    st.divider()
    st.header("📊 Risk Assessment Results")

    risk_level = "منخفضة" if proba < 0.3 else "متوسطة" if proba < 0.6 else "عالية"
    health_score = max(0, min(100, 100 - (proba * 100)))
    decision = "✅ APPROVED" if prediction == "Good" else "❌ REJECTED"
    decision_note = "منخفض المخاطرة - مناسب للموافقة" if prediction == "Good" else "مرتفع المخاطرة - يُنصح بإعادة التقييم"

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{proba*100:.1f}%", f"مستوى المخاطرة: {risk_level}")
    col2.metric("Financial Health", f"{health_score:.0f}/100")
    col3.metric("Final Decision", decision, decision_note)

    st.divider()
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.subheader("Factor Analysis")
        try:
            if model is not None and hasattr(model, "feature_importances_") and features is not None:
                importances = model.feature_importances_
                top_features = sorted(
                    [(features[i], importances[i]) for i in range(len(importances))],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f[1] for f in top_features],
                    y=[f[0][:20] for f in top_features],
                    orientation="h",
                    marker=dict(color=[f[1] for f in top_features], colorscale="Viridis", showscale=True),
                    text=[f"{f[1]*100:.1f}%" for f in top_features],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    xaxis=dict(title="Importance", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(title="", showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance غير متاح في وضع التقدير التقريبي.")
        except Exception as e:
            st.warning("تعذر عرض feature importances: " + str(e))

    with col_chart2:
        st.subheader("Decision Summary")
        st.write(f"**Payment-to-Income:** {(monthly_payment*12/annual_inc*100):.1f}%")
        st.write(f"**FICO Score:** {fico_score}")
        st.write(f"**Debt-to-Income:** {dti}%")
        st.write(f"**Interest Rate:** {int_rate}%")
        st.write(f"**Loan Grade:** {grade}")
        key_factor = "FICO Score" if fico_score < 650 else "Interest Rate" if int_rate > 15 else "Debt-to-Income"
        st.write(f"**Key Factor:** {key_factor}")

st.caption("© 2026 Smart Loan Advisor · Powered by AI · Data-driven Risk Assessment")
