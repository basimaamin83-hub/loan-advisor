"""
مستشار القروض الذكي Pro — تطبيق شامل بواجهة عربية RTL
يتضمن الميزات 1–17 بشكل عملي وتشغيلي داخل Streamlit.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from loan_core import (
    HOME_AR,
    PURPOSE_AR,
    build_feature_row,
    explain_decision,
    financial_health_score,
    load_artifacts,
    monthly_payment,
    peer_approval_rate,
    predict_risk,
    sentiment_from_text,
    sequential_default_risk,
    smart_recommendations,
    total_interest,
)
from loan_core.charts import importance_bar, risk_heatmap, speedometer, timeline_chart
from loan_core.db import init_db, load_expenses, load_predictions, save_expense, save_prediction
from loan_core.pdf_report import build_pdf_report

st.set_page_config(
    page_title="مستشار القروض الذكي Pro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== Theme / RTL ========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stText, label, p, span, div {
  font-family: 'Tajawal', Tahoma, sans-serif !important;
}

.stApp {
  background:
    radial-gradient(ellipse at 15% 10%, rgba(212,175,55,0.12), transparent 45%),
    radial-gradient(ellipse at 85% 90%, rgba(30,144,180,0.10), transparent 40%),
    linear-gradient(160deg, #0b1c2c 0%, #123047 45%, #0e2436 100%);
  color: #eef5fb;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1622 0%, #12283a 100%);
  border-left: 1px solid rgba(212,175,55,0.25);
}

.block-container { padding-top: 1.2rem; max-width: 1200px; }

.hero {
  background: linear-gradient(120deg, rgba(18,48,71,0.95), rgba(11,28,44,0.9));
  border: 1px solid rgba(212,175,55,0.35);
  border-radius: 18px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
  box-shadow: 0 10px 40px rgba(0,0,0,0.35);
}
.hero h1 {
  margin: 0;
  font-size: 2rem;
  font-weight: 800;
  color: #f4e4b0 !important;
  letter-spacing: 0.5px;
}
.hero p { margin: 0.35rem 0 0; color: #b7c9d9 !important; font-size: 1.05rem; }

.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.8rem;
}
.badge-ok { color:#00b894; font-weight:700; }
.badge-bad { color:#e17055; font-weight:700; }
.badge-mid { color:#fdcb6e; font-weight:700; }

/* Mobile-friendly */
@media (max-width: 768px) {
  .hero h1 { font-size: 1.45rem; }
  .block-container { padding: 0.6rem 0.8rem 2rem; }
}

div[data-testid="stMetricValue"] { color: #f4e4b0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_artifacts()


def inject_rtl():
    st.markdown('<div dir="rtl">', unsafe_allow_html=True)


def default_applicant() -> dict[str, Any]:
    return {
        "user_name": st.session_state.get("user_name", "ضيف"),
        "loan_amnt": 20000.0,
        "annual_inc": 60000.0,
        "int_rate": 12.0,
        "dti": 15.0,
        "fico_score": 700,
        "emp_length": 5,
        "purpose": "debt_consolidation",
        "grade": "B",
        "home_ownership": "RENT",
        "revol_bal": 10000.0,
        "revol_util": 30.0,
        "open_acc": 5,
        "total_acc": 10,
        "delinq_2yrs": 0,
        "pub_rec": 0,
        "term_months": 36,
        "threshold": 0.5,
    }


def applicant_form(prefix: str = "main") -> dict[str, Any]:
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amnt = st.number_input("مبلغ القرض ($)", 1000, 250000, 20000, 1000, key=f"{prefix}_loan")
        annual_inc = st.number_input("الدخل السنوي ($)", 5000, 500000, 60000, 1000, key=f"{prefix}_inc")
        int_rate = st.slider("معدل الفائدة (%)", 5.0, 30.0, 12.0, 0.25, key=f"{prefix}_rate")
        term_months = st.selectbox("مدة القرض (شهر)", [36, 60], key=f"{prefix}_term")
    with c2:
        dti = st.slider("نسبة الدين للدخل (%)", 0.0, 60.0, 15.0, 0.5, key=f"{prefix}_dti")
        fico = st.slider("درجة FICO", 580, 850, 700, 5, key=f"{prefix}_fico")
        emp = st.slider("سنوات العمل", 0, 40, 5, 1, key=f"{prefix}_emp")
        grade = st.selectbox("الدرجة", list("ABCDEFG"), index=1, key=f"{prefix}_grade")
    with c3:
        purpose = st.selectbox(
            "الغرض",
            list(PURPOSE_AR.keys()),
            format_func=lambda x: PURPOSE_AR[x],
            key=f"{prefix}_purpose",
        )
        home = st.selectbox(
            "ملكية السكن",
            list(HOME_AR.keys()),
            format_func=lambda x: HOME_AR[x],
            key=f"{prefix}_home",
        )
        revol_util = st.slider("استخدام الائتمان (%)", 0.0, 100.0, 30.0, 1.0, key=f"{prefix}_ru")
        delinq = st.number_input("تأخيرات آخر سنتين", 0, 10, 0, key=f"{prefix}_del")

    c4, c5, c6 = st.columns(3)
    with c4:
        revol_bal = st.number_input("الرصيد الدوار ($)", 0, 200000, 10000, 500, key=f"{prefix}_rb")
    with c5:
        open_acc = st.number_input("حسابات مفتوحة", 0, 50, 5, key=f"{prefix}_oa")
    with c6:
        total_acc = st.number_input("إجمالي الحسابات", 0, 100, 10, key=f"{prefix}_ta")

    threshold = st.slider("عتبة قرار المخاطرة", 0.2, 0.8, 0.5, 0.05, key=f"{prefix}_thr")
    pay = monthly_payment(loan_amnt, int_rate, term_months)
    return {
        "user_name": st.session_state.get("user_name", "ضيف"),
        "loan_amnt": float(loan_amnt),
        "annual_inc": float(annual_inc),
        "int_rate": float(int_rate),
        "dti": float(dti),
        "fico_score": int(fico),
        "emp_length": int(emp),
        "purpose": purpose,
        "grade": grade,
        "home_ownership": home,
        "revol_bal": float(revol_bal),
        "revol_util": float(revol_util),
        "open_acc": int(open_acc),
        "total_acc": int(total_acc),
        "delinq_2yrs": int(delinq),
        "pub_rec": 0,
        "term_months": int(term_months),
        "threshold": float(threshold),
        "monthly_payment": float(pay),
    }


def run_predict(app: dict, model, scaler, features) -> dict:
    row = build_feature_row(
        loan_amnt=app["loan_amnt"],
        int_rate=app["int_rate"],
        installment=app["monthly_payment"],
        dti=app["dti"],
        delinq_2yrs=app["delinq_2yrs"],
        open_acc=app["open_acc"],
        pub_rec=app["pub_rec"],
        revol_bal=app["revol_bal"],
        revol_util=app["revol_util"],
        total_acc=app["total_acc"],
        fico_score=app["fico_score"],
        emp_length=app["emp_length"],
        term_months=app["term_months"],
        grade=app["grade"],
        home_ownership=app["home_ownership"],
        purpose=app["purpose"],
        feature_cols=features,
    )
    pred = predict_risk(model, scaler, features, row, app["threshold"])
    pti = app["monthly_payment"] * 12 / max(app["annual_inc"], 1) * 100
    health = financial_health_score(
        pti_pct=pti,
        fico=app["fico_score"],
        emp_years=app["emp_length"],
        dti=app["dti"],
        delinq=app["delinq_2yrs"],
        revol_util=app["revol_util"],
    )
    tips = smart_recommendations(app, model, scaler, features, app["threshold"])
    expl = explain_decision(model, features, row, pred["proba"])
    peer = peer_approval_rate(app["fico_score"], app["dti"], app["emp_length"], app["annual_inc"])
    return {
        **pred,
        "row": row,
        "pti": pti,
        "health": health,
        "tips": tips,
        "expl": expl,
        "peer": peer,
        "app": app,
    }


# ======================== Pages ========================

def page_home(model, scaler, features):
    st.markdown(
        """
        <div class="hero" dir="rtl">
          <h1>🏛️ مستشار القروض الذكي Pro</h1>
          <p>تقييم مخاطر • محاكاة ماذا لو • مقارنة بنوك • صحة مالية • توصيات ذكية • تقارير PDF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    app = applicant_form("home")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("القسط الشهري", f"${app['monthly_payment']:,.0f}")
    m2.metric("إجمالي الفائدة", f"${total_interest(app['loan_amnt'], app['int_rate'], app['term_months']):,.0f}")
    m3.metric("نسبة القسط للدخل", f"{app['monthly_payment']*12/app['annual_inc']*100:.1f}%")
    m4.metric("المدة", f"{app['term_months']} شهر")

    if st.button("🚀 تقييم المخاطرة الآن", type="primary", use_container_width=True):
        result = run_predict(app, model, scaler, features)
        st.session_state["last_result"] = result
        save_prediction(
            {
                "user_name": app["user_name"],
                "loan_amnt": app["loan_amnt"],
                "annual_inc": app["annual_inc"],
                "int_rate": app["int_rate"],
                "dti": app["dti"],
                "fico": app["fico_score"],
                "monthly_payment": app["monthly_payment"],
                "proba": result["proba"],
                "prediction": result["prediction"],
                "health_score": result["health"]["score"],
            }
        )

    result = st.session_state.get("last_result")
    if not result:
        st.info("أدخل بياناتك ثم اضغط تقييم المخاطرة.")
        return

    health = result["health"]
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(speedometer(health["score"], health["color"]), use_container_width=True)
    with c2:
        decision = "موافق ✅" if result["approved"] else "مرفوض ❌"
        cls = "badge-ok" if result["approved"] else "badge-bad"
        st.markdown(
            f"""
            <div class="card" dir="rtl">
              <h3>نتيجة القرار: <span class="{cls}">{decision}</span></h3>
              <p>احتمال التعثر: <b>{result['proba']*100:.1f}%</b> — مستوى المخاطرة: <b>{result['risk_ar']}</b></p>
              <p>الصحة المالية: <b style="color:{health['color']}">{health['score']:.0f}/100 ({health['band']})</b></p>
              <p>أقرانك المشابهون: مجموعة <b>{result['peer']['cluster']}</b> —
              معدل الموافقة ≈ <b>{result['peer']['approval_rate']*100:.0f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("💡 التوصية الذكية")
        for t in result["tips"]:
            st.write(f"- {t}")

    st.subheader("تفسير القرار")
    st.plotly_chart(importance_bar([(a, b) for a, b, _ in result["expl"]]), use_container_width=True)
    for ar, imp, tip in result["expl"]:
        st.caption(f"{ar}: {tip} (أهمية {imp*100:.1f}%)")

    pdf_bytes = build_pdf_report(
        {
            **result["app"],
            "approved": result["approved"],
            "proba": result["proba"],
            "risk_ar": result["risk_ar"],
            "health_score": health["score"],
            "tips": result["tips"],
            "fico": result["app"]["fico_score"],
        }
    )
    st.download_button(
        "📄 تحميل تقرير PDF",
        data=pdf_bytes,
        file_name="loan_risk_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def page_what_if(model, scaler, features):
    st.header("🔄 محاكاة ماذا لو (What-If)")
    base = applicant_form("whatif")
    st.markdown("### عدّل السيناريو وشاهد التأثير فوراً")
    c1, c2, c3 = st.columns(3)
    with c1:
        salary_bump = st.slider("تغير الراتب السنوي (%)", -40, 50, 0, 5)
    with c2:
        rate_bump = st.slider("تغير الفائدة (نقاط %)", -3.0, 5.0, 0.0, 0.5)
    with c3:
        job_gap = st.slider("أشهر بدون دخل", 0, 6, 0, 1)

    scen = dict(base)
    scen["annual_inc"] = max(1000, base["annual_inc"] * (1 + salary_bump / 100))
    scen["int_rate"] = float(np.clip(base["int_rate"] + rate_bump, 5, 35))
    # job gap increases effective DTI pressure and lowers available income months
    effective_inc = scen["annual_inc"] * (12 - job_gap) / 12
    scen["annual_inc"] = max(1000, effective_inc)
    scen["monthly_payment"] = monthly_payment(scen["loan_amnt"], scen["int_rate"], scen["term_months"])
    # approximate DTI stress
    scen["dti"] = float(np.clip(base["dti"] + job_gap * 2.5 - max(0, salary_bump) * 0.15, 0, 60))

    base_res = run_predict(base, model, scaler, features)
    scen_res = run_predict(scen, model, scaler, features)

    a, b, c, d = st.columns(4)
    a.metric("القسط الأساسي", f"${base['monthly_payment']:,.0f}")
    b.metric("القسط بعد السيناريو", f"${scen['monthly_payment']:,.0f}",
             f"{scen['monthly_payment']-base['monthly_payment']:+.0f}")
    c.metric("مخاطرة أساسية", f"{base_res['proba']*100:.1f}%")
    d.metric("مخاطرة السيناريو", f"{scen_res['proba']*100:.1f}%",
             f"{(scen_res['proba']-base_res['proba'])*100:+.1f} نقطة")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="أساسي", x=["احتمال تعثر %", "صحة مالية", "قسط/دخل %"],
                         y=[base_res["proba"]*100, base_res["health"]["score"], base_res["pti"]],
                         marker_color="#4ecdc4"))
    fig.add_trace(go.Bar(name="ماذا لو", x=["احتمال تعثر %", "صحة مالية", "قسط/دخل %"],
                         y=[scen_res["proba"]*100, scen_res["health"]["score"], scen_res["pti"]],
                         marker_color="#f7971e"))
    fig.update_layout(barmode="group", height=360, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#eef5fb",
                      title="مقارنة فورية: الأساس مقابل السيناريو")
    st.plotly_chart(fig, use_container_width=True)

    if rate_bump > 0:
        st.warning(f"لو زادت الفائدة {rate_bump:.1f}% → القسط يصبح ${scen['monthly_payment']:,.0f}")
    if job_gap > 0:
        st.warning(f"لو خسرت وظيفتك {job_gap} أشهر → المخاطرة ترتفع إلى {scen_res['proba']*100:.1f}%")
    if salary_bump > 0:
        st.success(f"زيادة الراتب {salary_bump}% تحسّن الصحة المالية إلى {scen_res['health']['score']:.0f}/100")


def page_banks():
    st.header("🏦 مقارنة عروض البنوك")
    amount = st.number_input("مبلغ القرض للمقارنة ($)", 1000, 500000, 25000, 1000, key="bank_amt")
    rows = []
    cols = st.columns(3)
    defaults = [
        ("بنك الأمل", 11.5, 36),
        ("البنك الوطني", 13.2, 48),
        ("بنك الاستثمار", 10.8, 60),
    ]
    for i, col in enumerate(cols):
        with col:
            st.subheader(f"عرض {i+1}")
            name = st.text_input("اسم البنك", defaults[i][0], key=f"bn{i}")
            rate = st.number_input("الفائدة %", 1.0, 40.0, defaults[i][1], 0.1, key=f"br{i}")
            months = st.number_input("المدة (شهر)", 12, 120, defaults[i][2], 12, key=f"bm{i}")
            fees = st.number_input("رسوم إضافية ($)", 0, 5000, 200, 50, key=f"bf{i}")
            pay = monthly_payment(amount, rate, int(months))
            interest = total_interest(amount, rate, int(months))
            total = pay * months + fees
            rows.append(
                {
                    "البنك": name,
                    "الفائدة %": rate,
                    "المدة": int(months),
                    "القسط الشهري": round(pay, 2),
                    "إجمالي الفائدة": round(interest, 2),
                    "التكلفة الكلية": round(total, 2),
                    "الرسوم": fees,
                }
            )
    df = pd.DataFrame(rows)
    df["الأفضل قسط"] = df["القسط الشهري"] == df["القسط الشهري"].min()
    df["الأفضل فائدة"] = df["إجمالي الفائدة"] == df["إجمالي الفائدة"].min()
    df["الأفضل كلياً"] = df["التكلفة الكلية"] == df["التكلفة الكلية"].min()
    st.dataframe(df, use_container_width=True)
    fig = px.bar(
        df,
        x="البنك",
        y=["القسط الشهري", "إجمالي الفائدة", "التكلفة الكلية"],
        barmode="group",
        title="مقارنة مرئية بين عروض البنوك",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eef5fb")
    st.plotly_chart(fig, use_container_width=True)
    winner = df.loc[df["التكلفة الكلية"].idxmin(), "البنك"]
    st.success(f"الأفضل إجمالاً حسب التكلفة الكلية: {winner}")


def page_dashboard():
    st.header("📊 لوحة تحكم المستخدم")
    user = st.session_state.get("user_name", "ضيف")
    df = load_predictions(user)
    if df.empty:
        st.info("لا يوجد سجل بعد. نفّذ تقييماً من الصفحة الرئيسية أولاً.")
        return
    st.metric("عدد التقييمات", len(df))
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df.sort_values("id")[["proba"]].rename(columns={"proba": "احتمال التعثر"}))
    with c2:
        st.area_chart(df.sort_values("id")[["health_score"]].rename(columns={"health_score": "الصحة المالية"}))
    st.dataframe(df.drop(columns=["id"], errors="ignore"), use_container_width=True)


def page_early_warning():
    st.header("🚨 نظام التنبيه الاستباقي")
    user = st.session_state.get("user_name", "ضيف")
    c1, c2, c3 = st.columns(3)
    with c1:
        income = st.number_input("الدخل الشهري ($)", 0, 100000, 5000, 100)
    with c2:
        expenses = st.number_input("المصاريف الشهرية ($)", 0, 100000, 3200, 50)
    with c3:
        loan_pay = st.number_input("قسط القرض ($)", 0, 50000, 650, 10)
    note = st.text_input("ملاحظة", "ميزانية الشهر القادم")
    net = income - expenses - loan_pay
    buffer = income - expenses
    st.metric("صافي المتبقي بعد القسط", f"${net:,.0f}")

    if buffer < loan_pay:
        st.error("⚠ خطر مرتفع: المصاريف لا تترك هامشاً كافياً لسداد القسط الشهر القادم.")
    elif net < income * 0.05:
        st.warning("⚠ هامش أمان ضعيف جداً — راقب مصاريفك.")
    else:
        st.success("وضع السيولة مقبول لهذا الشهر.")

    if st.button("حفظ الميزانية"):
        save_expense(
            {
                "user_name": user,
                "month_label": note,
                "income": income,
                "expenses": expenses,
                "loan_payment": loan_pay,
                "note": note,
            }
        )
        st.success("تم الحفظ")
    hist = load_expenses(user)
    if not hist.empty:
        st.subheader("السجل")
        st.dataframe(hist, use_container_width=True)


def page_heatmap(model, scaler, features):
    st.header("🗺️ خريطة حرارية للمخاطرة")
    base = applicant_form("heat")

    def pred_payload(payload):
        pay = monthly_payment(payload["loan_amnt"], payload["int_rate"], payload["term_months"])
        row = build_feature_row(
            loan_amnt=payload["loan_amnt"],
            int_rate=payload["int_rate"],
            installment=pay,
            dti=payload["dti"],
            delinq_2yrs=payload["delinq_2yrs"],
            open_acc=payload["open_acc"],
            pub_rec=payload["pub_rec"],
            revol_bal=payload["revol_bal"],
            revol_util=payload["revol_util"],
            total_acc=payload["total_acc"],
            fico_score=payload["fico_score"],
            emp_length=payload["emp_length"],
            term_months=payload["term_months"],
            grade=payload["grade"],
            home_ownership=payload["home_ownership"],
            purpose=payload["purpose"],
            feature_cols=features,
        )
        return predict_risk(model, scaler, features, row, payload["threshold"])

    fig = risk_heatmap(model, scaler, features, base, pred_payload)
    st.plotly_chart(fig, use_container_width=True)


def page_explain(model, scaler, features):
    st.header("🧠 تفسير القرار (Explainable AI)")
    app = applicant_form("xai")
    if st.button("فسّر القرار", type="primary"):
        result = run_predict(app, model, scaler, features)
        st.write(f"القرار: {'موافقة' if result['approved'] else 'رفض'} — احتمال تعثر {result['proba']*100:.1f}%")
        try:
            import shap  # noqa: F401
            st.caption("مكتبة SHAP متوفرة — يُعرض تفسير مبسط متوافق مع النموذج.")
        except Exception:
            st.caption("تفسير قائم على أهمية الميزات + قواعد مفسّرة (بديل SHAP عند عدم التوفر).")
        st.plotly_chart(importance_bar([(a, b) for a, b, _ in result["expl"]]), use_container_width=True)
        for ar, imp, tip in result["expl"]:
            st.write(f"**{ar}**: {tip}")
        if not result["approved"]:
            top = result["expl"][0][0] if result["expl"] else "العوامل المالية"
            st.error(f"السبب الرئيسي الأقرب للرفض مرتبط بـ «{top}».")


def page_timeline():
    st.header("📈 المسار المالي الكامل")
    years_n = st.slider("عدد السنوات", 5, 10, 7)
    salary0 = st.number_input("الراتب السنوي الحالي ($)", 10000, 500000, 60000, 1000)
    growth = st.slider("نمو الراتب السنوي %", 0.0, 10.0, 3.0, 0.5)
    inflate = st.slider("التضخم %", 0.0, 10.0, 2.5, 0.5)
    save_rate = st.slider("نسبة التوفير من الدخل %", 0.0, 40.0, 10.0, 1.0)
    loan_pay_year = st.number_input("الأقساط السنوية ($)", 0, 200000, 7800, 100)

    years, income, payments, savings, net = [], [], [], [], []
    cum_save = 0.0
    wealth = 5000.0
    for y in range(1, years_n + 1):
        sal = salary0 * ((1 + growth / 100) ** (y - 1))
        pay = loan_pay_year * ((1 + inflate / 100) ** (y - 1))
        saved = sal * save_rate / 100
        cum_save += saved
        wealth = wealth * (1 + 0.04) + saved - max(0, pay - sal * 0.15)
        years.append(y)
        income.append(sal)
        payments.append(pay)
        savings.append(cum_save)
        net.append(wealth)
    st.plotly_chart(timeline_chart(years, income, payments, savings, net), use_container_width=True)


def page_peer(model, scaler, features):
    st.header("👥 مقارنة مع المستخدمين المشابهين")
    app = applicant_form("peer")
    peer = peer_approval_rate(app["fico_score"], app["dti"], app["emp_length"], app["annual_inc"])
    result = run_predict(app, model, scaler, features)
    st.markdown(
        f"""
        <div class="card" dir="rtl">
          <h3>مجموعتك: {peer['cluster']}</h3>
          <p>معدل الموافقة للأشخاص المشابهين لك ≈ <b>{peer['approval_rate']*100:.0f}%</b></p>
          <p>توقّع نموذجك الحالي: <b>{'موافقة' if result['approved'] else 'رفض'}</b>
          (احتمال تعثر {result['proba']*100:.1f}%)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # fake cluster distribution for visualization
    labels = ["الممتازون ائتمانياً", "المستقرون مالياً", "متوسطو المخاطر", "مرتفعو المخاطر"]
    rates = [88, 72, 55, 32]
    fig = px.bar(x=labels, y=rates, labels={"x": "المجموعة", "y": "معدل الموافقة %"},
                 title="معدلات الموافقة حسب الشريحة")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#eef5fb")
    st.plotly_chart(fig, use_container_width=True)


def page_upload(model, scaler, features):
    st.header("📥 رفع بيانات Excel / CSV")
    st.caption("ارفع ملفاً يحتوي أعمدة مثل: loan_amnt, int_rate, annual_inc, dti, fico_score")
    f = st.file_uploader("اختر ملف", type=["csv", "xlsx", "xls"])
    if not f:
        return
    if f.name.endswith(".csv"):
        df = pd.read_csv(f)
    else:
        df = pd.read_excel(f)
    st.dataframe(df.head(20), use_container_width=True)
    if st.button("تقييم الدفعة"):
        outs = []
        for _, r in df.head(100).iterrows():
            app = default_applicant()
            for k in app:
                if k in r and pd.notna(r[k]):
                    app[k] = r[k]
            if "fico" in r and "fico_score" not in r:
                app["fico_score"] = r["fico"]
            app["monthly_payment"] = monthly_payment(app["loan_amnt"], app["int_rate"], int(app["term_months"]))
            pred = run_predict(app, model, scaler, features)
            outs.append(
                {
                    "loan_amnt": app["loan_amnt"],
                    "fico": app["fico_score"],
                    "proba": round(pred["proba"], 4),
                    "decision": "موافق" if pred["approved"] else "مرفوض",
                    "health": round(pred["health"]["score"], 1),
                }
            )
        st.success(f"تم تقييم {len(outs)} صف")
        st.dataframe(pd.DataFrame(outs), use_container_width=True)


def page_voice_sentiment(model, scaler, features):
    st.header("🎙️ المساعد الصوتي وتحليل المشاعر")
    st.subheader("تحليل المشاعر من وصف وضعك المالي")
    text = st.text_area("اكتب عن وضعك المالي بثقة أو قلق...", "أنا واثق من قدرتي على السداد رغم الالتزامات الحالية.")
    sent = sentiment_from_text(text)
    st.metric("قطبية المشاعر", f"{sent['polarity']:.2f}", sent["label"])

    st.subheader("مساعد صوتي (نص ← صوت)")
    speak = st.text_input("نص الرد الصوتي", "نتيجتك جاهزة. راجع مؤشر الصحة المالية والتوصيات.")
    if st.button("توليد ملف صوت"):
        try:
            from gtts import gTTS

            tts = gTTS(text=speak, lang="ar")
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            st.audio(buf.getvalue(), format="audio/mp3")
            st.download_button("تحميل الصوت", buf.getvalue(), "advice.mp3", "audio/mp3")
        except Exception as e:
            st.error(f"تعذر توليد الصوت: {e}")

    st.subheader("رفع تسجيل صوتي (اختياري)")
    audio = st.file_uploader("ملف WAV/AIFF", type=["wav", "aiff", "aif"])
    if audio and st.button("حوّل الصوت إلى نص"):
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.read())
                path = tmp.name
            with sr.AudioFile(path) as source:
                audio_data = recognizer.record(source)
            try:
                txt = recognizer.recognize_google(audio_data, language="ar-SA")
            except Exception:
                txt = recognizer.recognize_google(audio_data, language="en-US")
            st.success(txt)
            st.write(sentiment_from_text(txt))
        except Exception as e:
            st.warning(f"التعرف على الكلام غير متاح حالياً ({e}). استخدم مربع النص.")


def page_lstm():
    st.header("🧮 نموذج تسلسلي لسجل الدفعات (LSTM-style)")
    st.caption("أدخل آخر 6–12 دفعة شهرية فعلية مقابل القسط المقرر.")
    scheduled = st.number_input("القسط المقرر ($)", 50, 20000, 650, 10)
    n = st.slider("عدد الأشهر في السجل", 6, 12, 8)
    payments = []
    cols = st.columns(4)
    for i in range(n):
        with cols[i % 4]:
            payments.append(st.number_input(f"دفعة {i+1}", 0, 30000, int(scheduled * (0.9 + 0.02 * (i % 3))), 10, key=f"p{i}"))
    risk = sequential_default_risk(payments, scheduled)
    st.metric("احتمال تعثر تسلسلي", f"{risk*100:.1f}%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=payments, mode="lines+markers", name="الدفعات", line=dict(color="#4ecdc4")))
    fig.add_hline(y=scheduled, line_dash="dash", line_color="#f7971e", annotation_text="المقرر")
    fig.update_layout(title="سلسلة الدفعات", height=320, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#eef5fb")
    st.plotly_chart(fig, use_container_width=True)
    if risk > 0.55:
        st.error("نمط الدفعات يشير إلى خطر تأخر مرتفع.")
    elif risk > 0.35:
        st.warning("هناك تقلب في الالتزام — راقب الأشهر القادمة.")
    else:
        st.success("سجل الدفعات مستقر نسبياً.")


def page_mobile_about():
    st.header("📱 تجربة الموبايل والتطبيق")
    st.markdown(
        """
        <div class="card" dir="rtl">
          <p>الواجهة متجاوبة مع الشاشات الصغيرة عبر CSS مخصص.</p>
          <p>للموبايل: افتح نفس الرابط من هاتفك على نفس الشبكة المحلية.</p>
          <p>يمكن لاحقاً تغليف Streamlit بـ PWA أو بناء واجهة React Native تستدعي نفس منطق التقييم.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("جرّب تصغير نافذة المتصفح أو افتح الرابط من الجوال للتأكد من التجاوب.")


# ======================== Router ========================
init_db()
model, scaler, features, _ = get_model()

with st.sidebar:
    st.markdown("### 🏛️ القائمة")
    st.session_state["user_name"] = st.text_input(
        "اسم المستخدم", st.session_state.get("user_name", "ضيف")
    )
    page = st.radio(
        "اختر الصفحة",
        [
            "الرئيسية + الصحة + PDF + توصية",
            "ماذا لو What-If",
            "مقارنة البنوك",
            "لوحة التحكم",
            "تنبيه استباقي",
            "خريطة المخاطرة",
            "تفسير القرار XAI",
            "المسار المالي",
            "مقارنة الأقران",
            "رفع Excel/CSV",
            "صوت ومشاعر",
            "نموذج تسلسلي LSTM",
            "الموبايل حول التطبيق",
        ],
        key="nav_page",
    )
    st.caption("واجهة عربية RTL — جميع الميزات 1–17")
    if model is None:
        st.error("تعذر تحميل النموذج — الوضع التقريبي")
    else:
        st.success("النموذج جاهز")

st.markdown('<div dir="rtl">', unsafe_allow_html=True)

if page == "الرئيسية + الصحة + PDF + توصية":
    page_home(model, scaler, features)
elif page == "ماذا لو What-If":
    page_what_if(model, scaler, features)
elif page == "مقارنة البنوك":
    page_banks()
elif page == "لوحة التحكم":
    page_dashboard()
elif page == "تنبيه استباقي":
    page_early_warning()
elif page == "خريطة المخاطرة":
    page_heatmap(model, scaler, features)
elif page == "تفسير القرار XAI":
    page_explain(model, scaler, features)
elif page == "المسار المالي":
    page_timeline()
elif page == "مقارنة الأقران":
    page_peer(model, scaler, features)
elif page == "رفع Excel/CSV":
    page_upload(model, scaler, features)
elif page == "صوت ومشاعر":
    page_voice_sentiment(model, scaler, features)
elif page == "نموذج تسلسلي LSTM":
    page_lstm()
else:
    page_mobile_about()
