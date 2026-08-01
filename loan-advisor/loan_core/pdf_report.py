"""PDF report generation (Latin-safe Helvetica)."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any


def build_pdf_report(data: dict[str, Any]) -> bytes:
    from fpdf import FPDF

    risk_map = {"منخفضة": "Low", "متوسطة": "Medium", "عالية": "High"}
    risk = data.get("risk_ar", "-")
    risk_en = risk_map.get(str(risk), "N/A")
    decision = "APPROVED" if data.get("approved") else "REJECTED"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(190, 10, "Smart Loan Advisor - Risk Report", align="C")
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(190, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")
    pdf.ln(14)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 8, "Applicant Summary")
    pdf.ln(9)
    pdf.set_font("Helvetica", "", 11)

    rows = [
        ("Name", str(data.get("user_name", "Guest"))),
        ("Loan Amount", f"${data.get('loan_amnt', 0):,.0f}"),
        ("Annual Income", f"${data.get('annual_inc', 0):,.0f}"),
        ("Interest Rate", f"{data.get('int_rate', 0):.2f}%"),
        ("DTI", f"{data.get('dti', 0):.1f}%"),
        ("FICO", f"{data.get('fico', 0):.0f}"),
        ("Monthly Payment", f"${data.get('monthly_payment', 0):,.2f}"),
        ("Grade", str(data.get("grade", "-"))),
        ("Purpose", str(data.get("purpose", "-"))),
    ]
    for label, value in rows:
        # strip non-latin safely
        value = value.encode("latin-1", "replace").decode("latin-1")
        pdf.cell(190, 7, f"{label}: {value}")
        pdf.ln(7)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 8, "Prediction Result")
    pdf.ln(9)
    pdf.set_font("Helvetica", "", 11)
    for line in [
        f"Decision: {decision}",
        f"Default Probability: {float(data.get('proba', 0))*100:.1f}%",
        f"Risk Level: {risk_en}",
        f"Financial Health Score: {float(data.get('health_score', 0)):.0f}/100",
    ]:
        pdf.cell(190, 7, line)
        pdf.ln(7)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 8, "Recommendations")
    pdf.ln(9)
    pdf.set_font("Helvetica", "", 11)
    for tip in [
        f"1. Default probability is {float(data.get('proba', 0))*100:.1f}%.",
        f"2. Health score is {float(data.get('health_score', 0)):.0f}/100.",
        "3. Consider lowering loan amount or extending term if rejected.",
        "4. Keep payment-to-income preferably under 30%.",
    ]:
        pdf.cell(190, 7, tip)
        pdf.ln(7)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        190,
        5,
        "Disclaimer: This report is an AI decision-support output and not a formal credit commitment.",
    )

    return bytes(pdf.output())
