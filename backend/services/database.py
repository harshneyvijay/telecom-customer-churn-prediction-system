import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "DB_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "data", "predictions.db")
)


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            churn_probability REAL NOT NULL,
            prediction INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            model_used TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(churn_probability, prediction, risk_level, model_used):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (churn_probability, prediction, risk_level, model_used, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            churn_probability,
            prediction,
            risk_level,
            model_used,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit=20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
