"""Plotly chart helpers."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8eef7", family="Tahoma"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def speedometer(score: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 36}},
            title={"text": "مؤشر الصحة المالية", "font": {"size": 18, "color": "#e8eef7"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8899aa"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(225,112,85,0.25)"},
                    {"range": [50, 70], "color": "rgba(253,203,110,0.25)"},
                    {"range": [70, 100], "color": "rgba(0,184,148,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#ffd200", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=280, **CHART_LAYOUT)
    return fig


def risk_heatmap(model, scaler, features, base_row: dict, predict_fn) -> go.Figure:
    ficos = list(range(580, 851, 20))
    dtis = list(range(5, 51, 5))
    z = []
    for dti in dtis:
        row_z = []
        for fico in ficos:
            payload = dict(base_row)
            payload["fico_score"] = fico
            payload["dti"] = dti
            row_z.append(predict_fn(payload)["proba"] * 100)
        z.append(row_z)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=ficos,
            y=dtis,
            colorscale="YlOrRd",
            colorbar=dict(title="احتمال %"),
            hovertemplate="FICO=%{x}<br>DTI=%{y}%<br>مخاطرة=%{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="خريطة حرارية للمخاطرة (FICO × نسبة الدين للدخل)",
        xaxis_title="درجة FICO",
        yaxis_title="DTI %",
        height=420,
        **CHART_LAYOUT,
    )
    return fig


def timeline_chart(years, income, payments, savings, net_worth) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=income, name="الدخل السنوي", line=dict(color="#4ecdc4", width=3)))
    fig.add_trace(go.Scatter(x=years, y=payments, name="الأقساط السنوية", line=dict(color="#e17055", width=3)))
    fig.add_trace(go.Scatter(x=years, y=savings, name="المدخرات التراكمية", line=dict(color="#fdcb6e", width=3)))
    fig.add_trace(go.Scatter(x=years, y=net_worth, name="صافي الثروة", line=dict(color="#a29bfe", width=3)))
    fig.update_layout(
        title="المسار المالي المتوقع",
        xaxis_title="السنة",
        yaxis_title="القيمة ($)",
        height=420,
        legend=dict(orientation="h", y=1.12),
        **CHART_LAYOUT,
    )
    return fig


def importance_bar(items: list[tuple[str, float]]) -> go.Figure:
    names = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color=vals, colorscale="Tealgrn"),
            text=[f"{v*100:.1f}%" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(title="أهمية العوامل", height=320, **CHART_LAYOUT)
    return fig


def compare_banks_style(df):
    """Return styler-friendly ranks: lower payment/interest is better."""
    return df
