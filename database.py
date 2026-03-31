"""
SQLite Database Setup and CRUD helpers
"""

import csv
import hashlib
import os
import secrets
import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent / "financial.db")
CSV_PATH = Path(__file__).parent / "ml" / "financial_data.csv"
CSV_COLUMNS = ["name", "username", "password", "income", "fixed_expenses",
               "variable_expenses", "total_expenses", "savings_goal", "lifestyle_score", "savings"]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


_SALT_SIZE = 32
_ITERATIONS = 260_000


def _hash(password: str, salt: bytes | None = None) -> str:
    """Returns 'salt_hex:hash_hex' using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = os.urandom(_SALT_SIZE)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return salt.hex() + ":" + dk.hex()


def _verify(password: str, stored: str) -> bool:
    """Constant-time verification against a stored PBKDF2 hash."""
    try:
        salt_hex, _ = stored.split(":")
        expected = _hash(password, bytes.fromhex(salt_hex))
        return secrets.compare_digest(expected, stored)
    except Exception:
        return False


def _migrate_users_columns(conn: sqlite3.Connection) -> None:
    """Add columns missing from older DBs (CREATE TABLE IF NOT EXISTS does not upgrade schema)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='users'"
    ).fetchone()
    if not row:
        return
    have = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    # (column_name, ALTER TABLE ... fragment after ADD COLUMN)
    upgrades = [
        ("name", "name TEXT NOT NULL DEFAULT ''"),
        ("token", "token TEXT"),
        ("income", "income REAL"),
        ("fixed_expenses", "fixed_expenses REAL"),
        ("variable_expenses", "variable_expenses REAL"),
        ("total_expenses", "total_expenses REAL"),
        ("savings_goal", "savings_goal REAL"),
        ("lifestyle_score", "lifestyle_score REAL"),
        ("savings", "savings REAL"),
        # ADD COLUMN: only literal constants allowed as DEFAULT in this SQLite build.
        ("created_at", "created_at TEXT"),
    ]
    for col, ddl in upgrades:
        if col not in have:
            conn.execute(f"ALTER TABLE users ADD COLUMN {ddl}")

    # Older schema used full_name; copy into name when name was added empty.
    cols_after = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    if "full_name" in cols_after and "name" in cols_after:
        conn.execute(
            """
            UPDATE users
            SET name = TRIM(full_name)
            WHERE TRIM(COALESCE(name, '')) = ''
              AND full_name IS NOT NULL
              AND TRIM(full_name) != ''
            """
        )


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                name              TEXT    NOT NULL DEFAULT '',
                username          TEXT    NOT NULL UNIQUE,
                password          TEXT    NOT NULL,
                token             TEXT,
                income            REAL,
                fixed_expenses    REAL,
                variable_expenses REAL,
                total_expenses    REAL,
                savings_goal      REAL,
                lifestyle_score   REAL,
                savings           REAL,
                created_at        TEXT DEFAULT (datetime('now'))
            )
        """)
        _migrate_users_columns(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('income','expense')),
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

        _create_default_categories(conn)


def _create_default_categories(conn: sqlite3.Connection) -> None:
    default = ["Food", "Travel", "Bills", "Groceries", "Rent", "Shopping"]
    for cat in default:
        try:
            conn.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (cat,))
        except sqlite3.Error:
            pass
    conn.commit()


def register_user(username: str, password: str, full_name: str = "") -> bool:
    """Returns True on success, False if username already exists."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (name, username, password) VALUES (?, ?, ?)",
                (full_name.strip(), username, _hash(password))
            )
            conn.commit()
        _append_registration_to_csv(username=username, full_name=full_name)
        return True
    except sqlite3.IntegrityError:
        return False


def login_user(username: str, password: str) -> str | None:
    """Returns a session token on success, None on failure."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, password FROM users WHERE username=?",
            (username,)
        ).fetchone()
        if not row or not _verify(password, row["password"]):
            return None
        token = secrets.token_hex(32)
        conn.execute("UPDATE users SET token=? WHERE id=?", (token, row["id"]))
        conn.commit()
    return token


def reset_password(username: str, new_password: str) -> bool:
    """Reset password for existing user (no OTP flow). Returns True if reset succeeded."""
    if not username.strip() or not new_password:
        return False

    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username=?",
            (username.strip(),)
        ).fetchone()
        if not row:
            return False

        conn.execute(
            "UPDATE users SET password=?, token=NULL WHERE id=?",
            (_hash(new_password), row["id"])
        )
        conn.commit()

    return True


def get_user_by_token(token: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT name, username FROM users WHERE token=?", (token,)
        ).fetchone()
    return dict(row) if row else None


def validate_token(token: str) -> bool:
    return get_user_by_token(token) is not None


def add_transaction(username: str, type_: str, category: str, amount: float, description: str = "") -> bool:
    if not username.strip() or type_ not in ('income', 'expense') or amount is None:
        return False
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO transactions (username, type, category, amount, description) VALUES (?, ?, ?, ?, ?)",
            (username.strip(), type_, category.strip(), float(amount), description.strip())
        )
        conn.commit()
    return True


def get_transactions(username: str | None = None, type_: str | None = None, category: str | None = None, limit: int = 100, offset: int = 0):
    query = "SELECT id, username, type, category, amount, description, created_at FROM transactions"
    clauses = []
    params = []
    if username:
        clauses.append("username = ?")
        params.append(username.strip())
    if type_ in ('income', 'expense'):
        clauses.append("type = ?")
        params.append(type_)
    if category:
        clauses.append("category = ?")
        params.append(category.strip())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def delete_transaction(txn_id: int) -> bool:
    with get_connection() as conn:
        result = conn.execute("DELETE FROM transactions WHERE id=?", (txn_id,))
        conn.commit()
    return result.rowcount > 0


def list_categories() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT id, name, created_at FROM categories ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def add_category(name: str) -> bool:
    if not name.strip():
        return False
    with get_connection() as conn:
        try:
            conn.execute("INSERT INTO categories (name) VALUES (?)", (name.strip(),))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def delete_category(name: str) -> bool:
    if not name.strip():
        return False
    with get_connection() as conn:
        result = conn.execute("DELETE FROM categories WHERE name=?", (name.strip(),))
        conn.commit()
    return result.rowcount > 0


def get_aggregate_stats() -> dict:
    with get_connection() as conn:
        users = conn.execute(
            """
            SELECT
              COUNT(*) AS total_users,
              SUM(COALESCE(income, 0)) AS total_income,
              SUM(COALESCE(fixed_expenses, 0) + COALESCE(variable_expenses, 0)) AS total_expenses,
              SUM(COALESCE(savings, 0)) AS total_savings
            FROM users
            """
        ).fetchone()
        recent = conn.execute(
            "SELECT id, username, type, category, amount, description, created_at FROM transactions ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        by_month = conn.execute(
            "SELECT strftime('%Y-%m', created_at) AS month, type, SUM(amount) AS amount FROM transactions GROUP BY month, type ORDER BY month DESC LIMIT 12"
        ).fetchall()
    return {
        "total_users": users["total_users"] or 0,
        "total_income": users["total_income"] or 0,
        "total_expenses": users["total_expenses"] or 0,
        "total_savings": users["total_savings"] or 0,
        "recent_transactions": [dict(r) for r in recent],
        "monthly_report": [dict(r) for r in by_month],
    }


def _append_to_csv(data: dict) -> None:
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "name": data.get("name", ""),
            "username": data.get("username", ""),
            "password": "[protected]",
            "income": data["income"],
            "fixed_expenses": data["fixed_expenses"],
            "variable_expenses": data["variable_expenses"],
            "total_expenses": data["total_expenses"],
            "savings_goal": data["savings_goal"],
            "lifestyle_score": data["lifestyle_score"],
            "savings": data.get("predicted_savings", 0),
        })


def _append_registration_to_csv(username: str, full_name: str = "") -> None:
    """Append a registration row to CSV without finance metrics."""
    username = (username or "").strip()
    if not username:
        return

    # Avoid duplicate registration entries for the same username.
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, fieldnames=CSV_COLUMNS)
            for row in reader:
                if (row.get("username") or "").strip() == username:
                    return

    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "name": (full_name or "").strip(),
            "username": (username or "").strip(),
            "password": "[protected]",
            "income": "",
            "fixed_expenses": "",
            "variable_expenses": "",
            "total_expenses": "",
            "savings_goal": "",
            "lifestyle_score": "",
            "savings": "",
        })


def insert_record(data: dict) -> int:
    username = (data.get("username") or "").strip()
    record_id = -1

    if username:
        with get_connection() as conn:
            result = conn.execute(
                """
                    UPDATE users SET
                        income=?, fixed_expenses=?, variable_expenses=?, total_expenses=?,
                        savings_goal=?, lifestyle_score=?, savings=?
                    WHERE username=?
                """,
                (
                    data["income"], data["fixed_expenses"], data["variable_expenses"],
                    data["total_expenses"], data["savings_goal"], data["lifestyle_score"],
                    data.get("predicted_savings"), username,
                ),
            )
            if result.rowcount == 0:
                # Create user record if not present (fallback for first inserted analysis)
                conn.execute(
                    "INSERT INTO users (username, name, income, fixed_expenses, variable_expenses, total_expenses, savings_goal, lifestyle_score, savings) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        username,
                        data.get("name", ""),
                        data["income"], data["fixed_expenses"], data["variable_expenses"],
                        data["total_expenses"], data["savings_goal"], data["lifestyle_score"],
                        data.get("predicted_savings"),
                    ),
                )
            cur = conn.execute("SELECT id FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            conn.commit()
            record_id = row["id"] if row else -1
    _append_to_csv(data)
    return record_id


def fetch_history(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT name, username, income, fixed_expenses, variable_expenses,
                      total_expenses, savings_goal, lifestyle_score, savings, created_at
               FROM users WHERE income IS NOT NULL
               ORDER BY created_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
