"""
Chart Generator: Matplotlib-based financial visualizations
Returns charts as base64-encoded PNG strings for frontend rendering
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def bar_chart(income: float, total_expenses: float, predicted_savings: float) -> str:
    """Income vs Expenses vs Savings bar chart"""
    fig, ax = plt.subplots(figsize=(7, 4))
    categories = ["Income", "Total Expenses", "Predicted Savings"]
    values = [income, total_expenses, predicted_savings]
    colors = ["#4CAF50", "#F44336", "#2196F3"]

    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + income * 0.01,
                f"Rs.{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Income vs Expenses vs Savings", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Amount (INR)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs.{x:,.0f}"))
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    return _fig_to_base64(fig)


def trend_chart(income: float, total_expenses: float, predicted_savings: float) -> str:
    """12-month cumulative savings trend based on actual income - expenses"""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Base monthly savings from actual income/expenses (not ML prediction)
    actual_monthly = (income - total_expenses) / 12

    # Realistic month-to-month variation seeded by income so it changes per user
    rng = np.random.default_rng(int(income) % (2**31))
    noise = rng.normal(0, abs(actual_monthly) * 0.12, 12)
    monthly_values = [actual_monthly + n for n in noise]
    cumulative = np.cumsum(monthly_values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, cumulative, marker="o", color="#2196F3", linewidth=2.5, markersize=6, label="Cumulative Savings")
    ax.fill_between(months, cumulative, alpha=0.12, color="#2196F3")

    # ML predicted annual savings as the goal line
    ax.axhline(y=predicted_savings, color="#4CAF50", linestyle="--", linewidth=1.5,
               label=f"ML Predicted: Rs.{predicted_savings:,.0f}")

    # Annotate each month's cumulative value
    for i, val in enumerate(cumulative):
        ax.annotate(f"Rs.{val:,.0f}", (months[i], val),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="#2196F3")

    ax.set_title("Projected Savings Trend (12 Months)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative Savings (INR)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rs.{x:,.0f}"))
    ax.legend(fontsize=8)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return _fig_to_base64(fig)


def pie_chart(fixed_expenses: float, variable_expenses: float, savings: float) -> str:
    """Expense + savings distribution pie chart"""
    labels = ["Fixed Expenses", "Variable Expenses", "Savings"]
    values = [fixed_expenses, variable_expenses, max(savings, 0)]
    colors = ["#FF7043", "#FFA726", "#66BB6A"]
    explode = (0.03, 0.03, 0.06)

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title("Income Distribution", fontsize=13, fontweight="bold", pad=12)
    fig.patch.set_facecolor("#ffffff")
    return _fig_to_base64(fig)


def generate_all_charts(income, fixed_expenses, variable_expenses,
                        total_expenses, predicted_savings) -> dict:
    return {
        "bar": bar_chart(income, total_expenses, predicted_savings),
        "trend": trend_chart(income, total_expenses, predicted_savings),
        "pie": pie_chart(fixed_expenses, variable_expenses, predicted_savings),
    }
