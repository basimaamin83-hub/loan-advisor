import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64

# ========================
# إعدادات الصفحة المتقدمة
# ========================
st.set_page_config(
    page_title="Smart Loan Advisor Pro",
    page_icon="🏛️",
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
# خلفية الصفحة (بنوك عالمية)
# ========================
# صورة خلفية متحركة (بنوك عالمية)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        position: relative;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(ellipse at 10% 20%, rgba(255, 215, 0, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 80%, rgba(0, 100, 255, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(100, 0, 255, 0.02) 0%, transparent 70%);
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.03; }
        50% { transform: translateY(-20px) rotate(5deg); opacity: 0.08; }
    }
    
    .bank-logos {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        z-index: -1;
        overflow: hidden;
        font-size: 80px;
        color: rgba(255, 255, 255, 0.02);
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        align-items: center;
        padding: 100px;
    }
    
    .bank-logos span {
        animation: float 20s infinite ease-in-out;
        display: inline-block;
        font-weight: 300;
        letter-spacing: 10px;
    }
</style>
<div class="bank-logos">
    <span>🏦 JP MORGAN 🏦 GOLDMAN SACHS 🏦 CITI 🏦 HSBC 🏦 UBS 🏦</span>
    <span>🏦 DEUTSCHE BANK 🏦 BARCLAYS 🏦 MORGAN STANLEY 🏦 BNP 🏦</span>
</div>
""", unsafe_allow_html=True)

# ========================
# تحميل النموذج
# ========================
@st.cache_resource
def load_model():
    model = joblib.load('models/loan_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/features.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    return model, scaler, features, label_encoders

model, scaler, features, label_encoders = load_model()

# ========================
# الهيدر الاحترافي
# ========================
st.markdown("""
<div class="header-container" style="text-align: center;">
    <h1 style="color: white; font-size: 3rem; margin: 0; font-weight: 700; text-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        🏛️ Smart Loan Advisor
    </h1>
    <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem; margin: 5px 0; letter-spacing: 2px; font-weight: 300;">
        AI-Powered Credit Risk Assessment Platform
    </p>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">
        <span style="color: rgba(255,215,0,0.6); font-size: 0.8rem;">✦ SECURE</span>
        <span style="color: rgba(255,215,0,0.6); font-size: 0.8rem;">✦ INTELLIGENT</span>
        <span style="color: rgba(255,215,0,0.6); font-size: 0.8rem;">✦ REAL-TIME</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ========================
# تخطيط الصفحة الرئيسية
# ========================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📋 Loan Application</p>', unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Financial Details</p>', unsafe_allow_html=True)
    
    monthly_payment = loan_amnt * (int_rate/100/12) * (1 + int_rate/100/12)**36 / ((1 + int_rate/100/12)**36 - 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${monthly_payment:,.0f}</div>
            <div class="metric-label">Monthly Payment</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{monthly_payment*12/annual_inc*100:.1f}%</div>
            <div class="metric-label">Payment-to-Income</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="margin: 20px 0;">', unsafe_allow_html=True)
if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
    grade_encoded = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}[grade]
    purpose_encoded = {
        "debt_consolidation": 0, "home_improvement": 1, "major_purchase": 2,
        "medical": 3, "car": 4, "credit_card": 5, "small_business": 6, "other": 7
    }[purpose]
    home_ownership_encoded = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}[home_ownership]
    term_encoded = 0
    
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
    
    st.markdown("---")
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    col_title1, col_title2, col_title3 = st.columns([1, 2, 1])
    with col_title2:
        st.markdown("""
        <h2 style="text-align: center; color: white; font-weight: 700; text-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            📊 Risk Assessment Results
        </h2>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_level = "منخفضة" if proba < 0.3 else "متوسطة" if proba < 0.6 else "عالية"
        color = "#00b894" if proba < 0.3 else "#fdcb6e" if proba < 0.6 else "#e17055"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {color};">
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1px;">Default Probability</div>
            <div style="font-size: 3.5rem; font-weight: 700; color: {color};">{proba*100:.1f}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {proba*100}%;"></div>
            </div>
            <div style="font-size: 1rem; margin-top: 10px; color: rgba(255,255,255,0.8);">مستوى المخاطرة: <strong style="color: {color};">{risk_level}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        health_score = max(0, min(100, 100 - (proba * 100)))
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #6c5ce7;">
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1px;">Financial Health</div>
            <div style="font-size: 3.5rem; font-weight: 700; color: #6c5ce7;">{health_score:.0f}<span style="font-size: 1.5rem;">/100</span></div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {health_score}%; background: linear-gradient(90deg, #6c5ce7, #a29bfe);"></div>
            </div>
            <div style="font-size: 1rem; margin-top: 10px; color: rgba(255,255,255,0.8);">الصحة المالية: <strong style="color: #6c5ce7;">{'ممتازة' if health_score > 70 else 'جيدة' if health_score > 50 else 'متوسطة'}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {'#00b894' if prediction == 'Good' else '#e17055'};">
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1px;">Final Decision</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: {'#00b894' if prediction == 'Good' else '#e17055'};">
                {'✅ APPROVED' if prediction == 'Good' else '❌ REJECTED'}
            </div>
            <div style="margin-top: 20px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 10px;">
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5);">القرض</div>
                <div style="font-size: 1.2rem; color: {'#00b894' if prediction == 'Good' else '#e17055'}; font-weight: 600;">
                    {'منخفض المخاطرة - مناسب للموافقة' if prediction == 'Good' else 'مرتفع المخاطرة - يُنصح بإعادة التقييم'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    col_chart1, col_chart2 = st.columns([3, 2])
    
    with col_chart1:
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 10px;">📊 Factor Analysis</p>', unsafe_allow_html=True)
        
        importances = model.feature_importances_
        top_features = [(features[i], importances[i]) for i in range(len(importances))]
        top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:10]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f[1] for f in top_features],
            y=[f[0][:20] for f in top_features],
            orientation='h',
            marker=dict(
                color=[f[1] for f in top_features],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f"{f[1]*100:.1f}%" for f in top_features],
            textposition='outside'
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(title='Importance', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='', showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 10px;">📋 Decision Summary</p>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border-radius: 15px; padding: 20px;">
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: rgba(255,255,255,0.6);">Payment-to-Income</span>
                <span style="color: white; font-weight: 600;">{(monthly_payment*12/annual_inc*100):.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: rgba(255,255,255,0.6);">FICO Score</span>
                <span style="color: white; font-weight: 600;">{fico_score}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: rgba(255,255,255,0.6);">Debt-to-Income</span>
                <span style="color: white; font-weight: 600;">{dti}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: rgba(255,255,255,0.6);">Interest Rate</span>
                <span style="color: white; font-weight: 600;">{int_rate}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: rgba(255,255,255,0.6);">Loan Grade</span>
                <span style="color: white; font-weight: 600;">{grade}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                <span style="color: rgba(255,255,255,0.6);">Key Factor</span>
                <span style="color: #ffd700; font-weight: 600;">{'FICO Score' if fico_score < 650 else 'Interest Rate' if int_rate > 15 else 'Debt-to-Income'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 20px; margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.05);">
    <p style="color: rgba(255,255,255,0.2); font-size: 0.8rem; letter-spacing: 2px;">
        © 2026 Smart Loan Advisor · Powered by AI · Data-driven Risk Assessment
    </p>
</div>
""", unsafe_allow_html=True)
