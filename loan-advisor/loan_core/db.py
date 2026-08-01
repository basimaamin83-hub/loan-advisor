"""SQLite persistence for prediction history."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

DB_PATH = os.path.join("data", "loan_history.db")


def _conn():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with _conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                user_name TEXT,
                loan_amnt REAL,
                annual_inc REAL,
                int_rate REAL,
                dti REAL,
                fico REAL,
                monthly_payment REAL,
                proba REAL,
                prediction TEXT,
                health_score REAL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                user_name TEXT,
                month_label TEXT,
                income REAL,
                expenses REAL,
                loan_payment REAL,
                note TEXT
            )
            """
        )


def save_prediction(row: dict[str, Any]):
    init_db()
    with _conn() as con:
        con.execute(
            """
            INSERT INTO predictions
            (created_at, user_name, loan_amnt, annual_inc, int_rate, dti, fico,
             monthly_payment, proba, prediction, health_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                row.get("user_name", "ضيف"),
                row.get("loan_amnt"),
                row.get("annual_inc"),
                row.get("int_rate"),
                row.get("dti"),
                row.get("fico"),
                row.get("monthly_payment"),
                row.get("proba"),
                row.get("prediction"),
                row.get("health_score"),
            ),
        )


def load_predictions(user_name: str | None = None) -> pd.DataFrame:
    init_db()
    with _conn() as con:
        if user_name:
            df = pd.read_sql_query(
                "SELECT * FROM predictions WHERE user_name = ? ORDER BY id DESC",
                con,
                params=(user_name,),
            )
        else:
            df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
    return df


def save_expense(row: dict[str, Any]):
    init_db()
    with _conn() as con:
        con.execute(
            """
            INSERT INTO expenses
            (created_at, user_name, month_label, income, expenses, loan_payment, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                row.get("user_name", "ضيف"),
                row.get("month_label", ""),
                row.get("income"),
                row.get("expenses"),
                row.get("loan_payment"),
                row.get("note", ""),
            ),
        )


def load_expenses(user_name: str | None = None) -> pd.DataFrame:
    init_db()
    with _conn() as con:
        if user_name:
            df = pd.read_sql_query(
                "SELECT * FROM expenses WHERE user_name = ? ORDER BY id DESC",
                con,
                params=(user_name,),
            )
        else:
            df = pd.read_sql_query("SELECT * FROM expenses ORDER BY id DESC", con)
    return df
