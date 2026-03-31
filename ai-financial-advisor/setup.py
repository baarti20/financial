"""
One-time setup: trains the ML model from existing financial_data.csv.
Run this before starting the server for the first time.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "ml"))

from model import load_data, train_and_evaluate

if __name__ == "__main__":
    print("Loading dataset from ml/financial_data.csv...")
    df = load_data()
    print(f"  [OK] Loaded {len(df):,} rows\n")

    print("Training ML models...")
    train_and_evaluate(df)
    print("\n[DONE] Setup complete! Run: uvicorn main:app --reload")
