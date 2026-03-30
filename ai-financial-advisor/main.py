"""
AI Financial Prediction & Advisory System - FastAPI Backend
Endpoints: /predict, /analyze, /advisor, /history, /charts
"""

import sys
from io import UnsupportedOperation

# Windows consoles often use cp1252; avoid UnicodeEncodeError on startup logs.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError, UnsupportedOperation):
        pass

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Resolve project root relative to this file so the app works from any cwd
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from advisor import full_analysis
from database import (
    fetch_history, init_db, insert_record,
    login_user, register_user, validate_token, get_user_by_token,
)

ML_MODEL_PATH = ROOT / "ml" / "model.pkl"
_ml_model = None


def _train_default_model():
    try:
        from ml.model import load_data, train_and_evaluate
        df = load_data()
        train_and_evaluate(df)
        return True
    except Exception as exc:
        print("[WARNING] Failed to train fallback model:", exc)
        return False


def _load_ml_model():
    global _ml_model
    if ML_MODEL_PATH.exists():
        try:
            with open(ML_MODEL_PATH, "rb") as f:
                _ml_model = pickle.load(f)
            print("Model loaded successfully.")
            return
        except Exception as exc:
            print("[WARNING] Could not load model from disk:", exc)
    # fallback train-only when the saved model is missing or invalid
    if _train_default_model() and ML_MODEL_PATH.exists():
        try:
            with open(ML_MODEL_PATH, "rb") as f:
                _ml_model = pickle.load(f)
            print("Fallback model trained and loaded.")
            return
        except Exception as exc:
            print("[WARNING] Fallback model training succeeded but loading failed:", exc)
    _ml_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _load_ml_model()
    yield


app = FastAPI(title="AI Financial Advisor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


# -- Auth Schemas -------------------------------------------------------------

class AuthInput(BaseModel):
    username: str
    password: str
    full_name: str = ""


# -- Schemas ------------------------------------------------------------------

class FinancialInput(BaseModel):
    income: float = Field(..., gt=0, description="Annual income in INR")
    fixed_expenses: float = Field(..., ge=0)
    variable_expenses: float = Field(..., ge=0)
    savings_goal: float = Field(..., ge=0)
    lifestyle_score: float = Field(..., ge=1, le=10)

    @property
    def total_expenses(self) -> float:
        return self.fixed_expenses + self.variable_expenses


# -- Helper -------------------------------------------------------------------

def _predict(data: FinancialInput) -> float:
    total = data.fixed_expenses + data.variable_expenses
    if _ml_model is not None:
        import pandas as pd
        X = pd.DataFrame([{
            "income": data.income,
            "total_expenses": total,
            "savings_goal": data.savings_goal,
            "lifestyle_score": data.lifestyle_score,
        }])
        return round(float(_ml_model.predict(X)[0]), 2)
    return round(data.income - total, 2)


# -- Endpoints ----------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root(request: Request):
    token = request.cookies.get("session")
    if not token or not validate_token(token):
        return RedirectResponse("/login")
    return FileResponse(str(ROOT / "static" / "index.html"))


@app.get("/login", include_in_schema=False)
async def login_page():
    return FileResponse(str(ROOT / "static" / "login.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# -- Auth Endpoints -----------------------------------------------------------

@app.post("/auth/register")
async def register(data: AuthInput):
    if not data.username.strip() or not data.password.strip():
        return JSONResponse({"error": "Username and password required"}, status_code=400)
    ok = register_user(data.username.strip(), data.password, data.full_name)
    if not ok:
        return JSONResponse({"error": "Username already exists"}, status_code=409)
    return {"message": "Registered successfully"}


@app.post("/auth/login")
async def login(data: AuthInput):
    token = login_user(data.username.strip(), data.password)
    if not token:
        return JSONResponse({"error": "Invalid username or password"}, status_code=401)
    response = JSONResponse({"message": "Login successful"})
    response.set_cookie("session", token, httponly=True, samesite="lax")
    return response


@app.post("/auth/logout")
async def logout():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("session")
    return response


@app.get("/auth/me")
async def auth_me(request: Request):
    """Current user from session cookie (for navbar)."""
    token = request.cookies.get("session", "")
    user = get_user_by_token(token)
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    name = (user.get("name") or "").strip()
    username = user.get("username") or ""
    display = name or username or "User"
    return {"name": name, "username": username, "display_name": display}


@app.post("/predict")
async def predict(data: FinancialInput):
    """Predict savings using: Income − (Fixed Expenses + Variable Expenses)."""
    predicted_savings = _predict(data)
    return {
        "predicted_savings": predicted_savings,
        "actual_savings_estimate": round(data.income - data.total_expenses, 2),
    }


@app.post("/analyze")
async def analyze(data: FinancialInput):
    """Return overspending risk and financial health score."""
    total = data.fixed_expenses + data.variable_expenses
    predicted_savings = _predict(data)
    analysis = full_analysis(data.income, total, data.savings_goal, data.lifestyle_score, predicted_savings)
    return {
        "predicted_savings": predicted_savings,
        "health_score": analysis["health_score"],
        "overspending": analysis["overspending"],
    }


@app.post("/advisor")
async def advisor(request: Request, data: FinancialInput):
    """Return full analysis + personalized AI advice."""
    token = request.cookies.get("session", "")
    user = get_user_by_token(token) or {"username": "", "name": ""}

    total = data.fixed_expenses + data.variable_expenses
    predicted_savings = _predict(data)
    analysis = full_analysis(data.income, total, data.savings_goal, data.lifestyle_score, predicted_savings)

    record_id = insert_record({
        "username": user["username"],
        "name": user["name"],
        "income": data.income,
        "fixed_expenses": data.fixed_expenses,
        "variable_expenses": data.variable_expenses,
        "total_expenses": total,
        "savings_goal": data.savings_goal,
        "lifestyle_score": data.lifestyle_score,
        "predicted_savings": predicted_savings,
        "health_score": analysis["health_score"],
    })

    return {
        "record_id": record_id,
        "predicted_savings": predicted_savings,
        "health_score": analysis["health_score"],
        "overspending": analysis["overspending"],
        "advice": analysis["advice"],
    }


@app.post("/charts")
async def charts(data: FinancialInput):
    """Return raw chart data for frontend rendering."""
    total = data.fixed_expenses + data.variable_expenses
    predicted_savings = _predict(data)
    monthly_savings = predicted_savings / 12
    cumulative = [round(monthly_savings * (i + 1), 2) for i in range(12)]
    return {
        "bar": {
            "labels": ["Income", "Total Expenses", "Predicted Savings"],
            "values": [data.income, total, predicted_savings],
        },
        "trend": {
            "labels": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            "cumulative": cumulative,
            "predicted_savings": predicted_savings,
        },
        "pie": {
            "labels": ["Fixed Expenses", "Variable Expenses", "Savings"],
            "values": [data.fixed_expenses, data.variable_expenses, max(predicted_savings, 0)],
        },
    }


@app.get("/history")
async def history(limit: int = 20):
    """Retrieve recent financial records from the database."""
    records = fetch_history(limit)
    return {"count": len(records), "records": records}
