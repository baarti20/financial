"""
Synthetic Financial Dataset Generator
Generates 100,000 rows of realistic financial data
"""

import numpy as np
import pandas as pd

def generate_dataset(n=100_000, seed=42):
    rng = np.random.default_rng(seed)

    # Income: Rs.2L–Rs.50L annually (Indian salary range)
    income = rng.uniform(200_000, 5_000_000, n)

    # Expenses: 50–90% of income
    expense_ratio = rng.uniform(0.50, 0.90, n)
    total_expenses = income * expense_ratio

    # Fixed expenses: 40–65% of total expenses (rent, EMI, insurance)
    fixed_ratio = rng.uniform(0.40, 0.65, n)
    fixed_expenses = total_expenses * fixed_ratio
    variable_expenses = total_expenses - fixed_expenses

    # Savings goal: 10–40% of income
    savings_goal = income * rng.uniform(0.10, 0.40, n)

    # Lifestyle score: 1–10 (higher = more spending on lifestyle)
    lifestyle_score = np.clip(
        rng.normal(loc=5, scale=2, size=n), 1, 10
    ).round(1)

    # Target: actual savings
    savings = income - total_expenses

    df = pd.DataFrame({
        "income": income.round(2),
        "fixed_expenses": fixed_expenses.round(2),
        "variable_expenses": variable_expenses.round(2),
        "total_expenses": total_expenses.round(2),
        "savings_goal": savings_goal.round(2),
        "lifestyle_score": lifestyle_score,
        "savings": savings.round(2),
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("ml/financial_data.csv", index=False)
    print(f"Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.describe().round(2))
