"""
ML Model Training: Linear Regression + Random Forest
Trains, evaluates, and saves the best model using pickle
"""

import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import generate_dataset

ROOT = Path(__file__).parent
FEATURES = ["income", "total_expenses", "savings_goal", "lifestyle_score"]
TARGET = "savings"
MODEL_PATH = str(ROOT / "model.pkl")


def load_data():
    csv_path = ROOT / "financial_data.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Dataset not found - generating...")
        df = generate_dataset()
        df.to_csv(csv_path, index=False)
    return df[FEATURES + [TARGET]].dropna()


def train_and_evaluate(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
        }
        print(f"{name} | R2: {results[name]['r2']:.4f} | MAE: ${results[name]['mae']:,.2f}")

    # Save the best model (Random Forest)
    best = results["Random Forest"]["model"]
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best, f)
    print(f"\nModel saved to {MODEL_PATH}")
    return results


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    df = load_data()
    train_and_evaluate(df)
