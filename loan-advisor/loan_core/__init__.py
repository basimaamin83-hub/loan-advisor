"""Core helpers for Smart Loan Advisor Pro."""
from __future__ import annotations

import os
import traceback
from typing import Any

import joblib
import numpy as np
import pandas as pd

FEATURE_NAMES_AR = {
    "loan_amnt": "مبلغ القرض",
    "int_rate": "معدل الفائدة",
    "installment": "القسط الشهري",
    "dti": "نسبة الدين للدخل",
    "delinq_2yrs": "التأخيرات (سنتين)",
    "open_acc": "الحسابات المفتوحة",
    "pub_rec": "السجلات العامة",
    "revol_bal": "الرصيد الدوار",
    "revol_util": "استخدام الائتمان",
    "total_acc": "إجمالي الحسابات",
    "fico_score": "درجة FICO",
    "emp_length_clean": "سنوات العمل",
    "term_encoded": "مدة القرض",
    "grade_encoded": "الدرجة الائتمانية",
    "home_ownership_encoded": "ملكية السكن",
    "verification_status_encoded": "حالة التحقق",
    "purpose_encoded": "الغرض",
}

GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
PURPOSE_MAP = {
    "debt_consolidation": 0,
    "home_improvement": 1,
    "major_purchase": 2,
    "medical": 3,
    "car": 4,
    "credit_card": 5,
    "small_business": 6,
    "other": 7,
}
HOME_MAP = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}
PURPOSE_AR = {
    "debt_consolidation": "توحيد ديون",
    "home_improvement": "تحسين منزل",
    "major_purchase": "شراء كبير",
    "medical": "طبي",
    "car": "سيارة",
    "credit_card": "بطاقة ائتمان",
    "small_business": "مشروع صغير",
    "other": "أخرى",
}
HOME_AR = {"RENT": "إيجار", "OWN": "ملك", "MORTGAGE": "رهن"}


def monthly_payment(principal: float, annual_rate_pct: float, months: int = 36) -> float:
    r = annual_rate_pct / 100 / 12
    if r <= 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def total_interest(principal: float, annual_rate_pct: float, months: int = 36) -> float:
    pay = monthly_payment(principal, annual_rate_pct, months)
    return pay * months - principal


def load_artifacts():
    paths = {
        "model": os.path.join("models", "loan_model.pkl"),
        "scaler": os.path.join("models", "scaler.pkl"),
        "features": os.path.join("models", "features.pkl"),
        "encoders": os.path.join("models", "label_encoders.pkl"),
    }
    for p in paths.values():
        if not os.path.exists(p):
            return None, None, None, None
    try:
        return (
            joblib.load(paths["model"]),
            joblib.load(paths["scaler"]),
            joblib.load(paths["features"]),
            joblib.load(paths["encoders"]),
        )
    except Exception:
        traceback.print_exc()
        return None, None, None, None


def build_feature_row(
    *,
    loan_amnt,
    int_rate,
    installment,
    dti,
    delinq_2yrs,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    fico_score,
    emp_length,
    term_months=36,
    grade="B",
    home_ownership="RENT",
    verification_status=0,
    purpose="debt_consolidation",
    feature_cols=None,
) -> pd.DataFrame:
    row = {
        "loan_amnt": float(loan_amnt),
        "int_rate": float(int_rate),
        "installment": float(installment),
        "dti": float(dti),
        "delinq_2yrs": float(delinq_2yrs),
        "open_acc": float(open_acc),
        "pub_rec": float(pub_rec),
        "revol_bal": float(revol_bal),
        "revol_util": float(revol_util),
        "total_acc": float(total_acc),
        "fico_score": float(fico_score),
        "emp_length_clean": float(emp_length),
        "term_encoded": 0 if term_months <= 36 else 1,
        "grade_encoded": GRADE_MAP.get(grade, 1),
        "home_ownership_encoded": HOME_MAP.get(home_ownership, 0),
        "verification_status_encoded": float(verification_status),
        "purpose_encoded": PURPOSE_MAP.get(purpose, 0),
    }
    cols = list(feature_cols) if feature_cols is not None else list(row.keys())
    return pd.DataFrame([[row[c] for c in cols]], columns=cols)


def heuristic_proba(fico, dti, int_rate, emp_length=5, pti=20) -> float:
    score = 0.0
    score += np.clip((720 - fico) / 200, -0.2, 0.45)
    score += np.clip((dti - 20) / 40, -0.1, 0.35)
    score += np.clip((int_rate - 12) / 20, -0.05, 0.25)
    score += np.clip((pti - 25) / 40, -0.05, 0.2)
    score -= np.clip(emp_length / 40, 0, 0.1)
    return float(np.clip(0.35 + score, 0.02, 0.98))


def predict_risk(model, scaler, features, row_df: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    if model is None or scaler is None or features is None:
        vals = row_df.iloc[0]
        proba = heuristic_proba(
            vals.get("fico_score", 700),
            vals.get("dti", 15),
            vals.get("int_rate", 12),
            vals.get("emp_length_clean", 5),
        )
    else:
        scaled = scaler.transform(row_df[list(features)])
        proba = float(model.predict_proba(scaled)[0][1])
    prediction = "Bad" if proba > threshold else "Good"
    return {
        "proba": proba,
        "prediction": prediction,
        "approved": prediction == "Good",
        "risk_ar": "منخفضة" if proba < 0.3 else "متوسطة" if proba < 0.6 else "عالية",
    }


def financial_health_score(
    *,
    pti_pct: float,
    fico: float,
    emp_years: float,
    dti: float,
    delinq: float = 0,
    revol_util: float = 30,
) -> dict[str, Any]:
    """Composite 0-100 readiness score with color band."""
    fico_part = np.clip((fico - 580) / (850 - 580) * 35, 0, 35)
    pti_part = np.clip(25 - abs(pti_pct - 15) * 0.8, 0, 25)
    emp_part = np.clip(emp_years / 15 * 15, 0, 15)
    dti_part = np.clip((40 - dti) / 40 * 15, 0, 15)
    clean_part = np.clip(10 - delinq * 3 - max(0, revol_util - 50) / 20, 0, 10)
    score = float(np.clip(fico_part + pti_part + emp_part + dti_part + clean_part, 0, 100))
    if score >= 70:
        band, color = "ممتازة", "#00b894"
    elif score >= 50:
        band, color = "جيدة", "#fdcb6e"
    else:
        band, color = "ضعيفة", "#e17055"
    return {"score": score, "band": band, "color": color}


def explain_decision(model, features, row_df: pd.DataFrame, proba: float) -> list[tuple[str, float, str]]:
    """Lightweight explainability without SHAP dependency."""
    explanations = []
    if model is not None and hasattr(model, "feature_importances_") and features is not None:
        imps = model.feature_importances_
        vals = row_df.iloc[0]
        ranked = sorted(zip(features, imps), key=lambda x: x[1], reverse=True)[:6]
        for name, imp in ranked:
            ar = FEATURE_NAMES_AR.get(name, name)
            v = vals[name]
            if name == "dti":
                tip = "مرتفعة عن المعتاد" if v > 35 else "ضمن المعدل"
            elif name == "fico_score":
                tip = "منخفضة" if v < 660 else "جيدة"
            elif name == "int_rate":
                tip = "مرتفعة" if v > 15 else "معتدلة"
            elif name == "revol_util":
                tip = "استخدام ائتمان مرتفع" if v > 50 else "استخدام معتدل"
            else:
                tip = "عامل مؤثر"
            direction = "يزيد المخاطرة" if proba > 0.5 else "يدعم الموافقة"
            explanations.append((ar, float(imp), f"{tip} — {direction}"))
    else:
        explanations = [
            ("درجة FICO", 0.25, "عامل رئيسي في التقييم"),
            ("نسبة الدين للدخل", 0.2, "عامل رئيسي في التقييم"),
            ("معدل الفائدة", 0.15, "عامل رئيسي في التقييم"),
        ]
    return explanations


def smart_recommendations(base: dict, model, scaler, features, threshold=0.5) -> list[str]:
    """Try alternate loan settings to improve approval odds."""
    tips = []
    amounts = [base["loan_amnt"] * 0.75, base["loan_amnt"] * 0.5, 10000]
    terms = [36, 60]
    rates = [max(5, base["int_rate"] - 2), base["int_rate"]]

    best = None
    for amnt in amounts:
        for term in terms:
            for rate in rates:
                pay = monthly_payment(amnt, rate, term)
                row = build_feature_row(
                    loan_amnt=amnt,
                    int_rate=rate,
                    installment=pay,
                    dti=base["dti"],
                    delinq_2yrs=base["delinq_2yrs"],
                    open_acc=base["open_acc"],
                    pub_rec=base["pub_rec"],
                    revol_bal=base["revol_bal"],
                    revol_util=base["revol_util"],
                    total_acc=base["total_acc"],
                    fico_score=base["fico_score"],
                    emp_length=base["emp_length"],
                    term_months=term,
                    grade=base["grade"],
                    home_ownership=base["home_ownership"],
                    purpose=base["purpose"],
                    feature_cols=features,
                )
                pred = predict_risk(model, scaler, features, row, threshold)
                cost = total_interest(amnt, rate, term)
                if pred["approved"]:
                    cand = (cost, amnt, term, rate, pred["proba"])
                    if best is None or cand[0] < best[0]:
                        best = cand

    if best:
        _, amnt, term, rate, proba = best
        tips.append(
            f"أنصحك بمبلغ حوالي ${amnt:,.0f} لمدة {term} شهراً بفائدة ~{rate:.1f}% "
            f"(احتمال تعثر ≈ {proba*100:.1f}%) لتحسين فرص الموافقة بأقل تكلفة فائدة."
        )
    else:
        if base["fico_score"] < 680:
            tips.append("حسّن درجة FICO فوق 680 قبل إعادة التقديم.")
        if base["dti"] > 30:
            tips.append("اخفض نسبة الدين للدخل تحت 30% عبر تقليل الالتزامات.")
        tips.append("جرّب تقليل مبلغ القرض بنسبة 40–50% أو تمديد المدة إلى 60 شهراً.")

    pti = monthly_payment(base["loan_amnt"], base["int_rate"], 36) * 12 / max(base["annual_inc"], 1) * 100
    if pti > 35:
        tips.append(f"نسبة القسط للدخل حالياً {pti:.1f}% — المفضل إبقاؤها دون 30%.")
    return tips


def peer_approval_rate(fico: float, dti: float, emp: float, income: float) -> dict[str, Any]:
    """Synthetic peer clusters via simple rules mimicking K-Means segments."""
    # cluster id based on normalized features
    f = (fico - 580) / 270
    d = 1 - min(dti, 50) / 50
    e = min(emp, 20) / 20
    i = min(income, 200000) / 200000
    centroid = 0.4 * f + 0.25 * d + 0.2 * e + 0.15 * i
    if centroid >= 0.7:
        cluster, rate = "الممتازون ائتمانياً", 0.88
    elif centroid >= 0.5:
        cluster, rate = "المستقرون مالياً", 0.72
    elif centroid >= 0.35:
        cluster, rate = "متوسطو المخاطر", 0.55
    else:
        cluster, rate = "مرتفعو المخاطر", 0.32
    return {"cluster": cluster, "approval_rate": rate, "centroid": centroid}


def sequential_default_risk(payment_history: list[float], scheduled: float) -> float:
    """Lightweight sequence model (LSTM-style) over payment ratios."""
    if not payment_history:
        return 0.4
    ratios = np.array([p / max(scheduled, 1) for p in payment_history], dtype=float)
    # exponential weighted shortfall + volatility
    weights = np.exp(np.linspace(-1, 0, len(ratios)))
    weights /= weights.sum()
    shortfall = np.clip(1 - ratios, 0, 1)
    ew_short = float(np.dot(weights, shortfall))
    vol = float(np.std(ratios)) if len(ratios) > 1 else 0.0
    late_streak = 0
    for r in ratios[::-1]:
        if r < 0.95:
            late_streak += 1
        else:
            break
    risk = np.clip(0.15 + 0.55 * ew_short + 0.25 * vol + 0.05 * late_streak, 0.02, 0.98)
    return float(risk)


def sentiment_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"polarity": 0.0, "label": "محايد", "note": "لم يُدخل نص"}
    try:
        from textblob import TextBlob

        # Arabic-heavy text: use keyword heuristic + TextBlob for Latin parts
        polarity = float(TextBlob(text).sentiment.polarity)
    except Exception:
        polarity = 0.0

    neg_kw = ["خايف", "قلق", "صعب", "دين", "متأخر", "أزمة", "ضايق", "مش قادر", "afraid", "worried", "debt", "late"]
    pos_kw = ["واثق", "مستقر", "قادر", "مرتاح", "ممتاز", "جيد", "confident", "stable", "secure", "fine"]
    low = text.lower()
    polarity += 0.15 * sum(1 for k in pos_kw if k in low or k in text)
    polarity -= 0.15 * sum(1 for k in neg_kw if k in low or k in text)
    polarity = float(np.clip(polarity, -1, 1))
    if polarity > 0.2:
        label = "إيجابي — ثقة بالسداد"
    elif polarity < -0.2:
        label = "سلبي — قلق من السداد"
    else:
        label = "محايد"
    return {"polarity": polarity, "label": label, "note": "تحليل مشاعر نصّي"}
