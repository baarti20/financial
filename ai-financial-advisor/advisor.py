"""
AI Advisor: Rule-based financial analysis engine
Provides health scores, overspending detection, and personalized advice
"""


def compute_health_score(income: float, predicted_savings: float) -> float:
    """Health Score = (Savings / Income) × 100"""
    if income <= 0:
        return 0.0
    return round(min(100, max(0, (predicted_savings / income) * 100)), 1)


def detect_overspending(income: float, total_expenses: float) -> dict:
    ratio = total_expenses / income if income > 0 else 1
    if ratio >= 0.90:
        level, label = "critical", "Critical Overspending"
    elif ratio >= 0.80:
        level, label = "high", "High Overspending Risk"
    elif ratio >= 0.70:
        level, label = "moderate", "Moderate Spending"
    else:
        level, label = "healthy", "Healthy Spending"
    return {"ratio": round(ratio, 4), "level": level, "label": label}


def generate_advice(income: float, total_expenses: float,
                    savings_goal: float, lifestyle_score: float,
                    predicted_savings: float) -> list[str]:
    advice = []
    ratio = total_expenses / income if income > 0 else 1
    savings_rate = (income - total_expenses) / income if income > 0 else 0
    goal_gap = savings_goal - predicted_savings

    # Spending ratio advice
    if ratio >= 0.90:
        advice.append("\U0001f6a8 You are spending over 90% of your income. Immediate budget review required.")
    elif ratio >= 0.80:
        advice.append("\u26a0\ufe0f Expenses exceed 80% of income. Cut discretionary spending to avoid debt.")
    elif ratio >= 0.70:
        advice.append("\U0001f4ca Spending is at 70-80% of income. You're on track but monitor variable costs.")
    else:
        advice.append("\u2705 Great job! Your spending ratio is healthy (below 70%).")

    # Savings goal advice
    if goal_gap > 0:
        advice.append(f"\U0001f4a1 You're Rs.{goal_gap:,.0f} short of your savings goal. Consider automating savings transfers.")
    else:
        advice.append(f"\U0001f3af You're exceeding your savings goal by Rs.{abs(goal_gap):,.0f}. Consider investing the surplus.")

    # Savings rate advice
    if savings_rate < 0.10:
        advice.append("\U0001f4c9 Savings rate below 10%. Aim for at least 20% using the 50/30/20 rule.")
    elif savings_rate < 0.20:
        advice.append("\U0001f4c8 Savings rate is 10-20%. Push toward 20% for long-term financial security.")
    else:
        advice.append("\U0001f4b0 Excellent savings rate! Explore mutual funds, SIPs, or FDs for the surplus.")

    # Lifestyle score advice
    if lifestyle_score >= 8:
        advice.append("\U0001f6cd\ufe0f High lifestyle score detected. Review OTT subscriptions, dining out, and shopping costs.")
    elif lifestyle_score >= 6:
        advice.append("\u2615 Moderate lifestyle spending. Small cuts (e.g., cooking at home) can save Rs.2,000-Rs.5,000/month.")
    else:
        advice.append("\U0001f331 Low lifestyle spending. Reward yourself occasionally - balance is key.")

    # Emergency fund check
    monthly_expenses = total_expenses / 12
    emergency_fund_needed = monthly_expenses * 6
    advice.append(f"\U0001f3e6 Recommended emergency fund: Rs.{emergency_fund_needed:,.0f} (6 months of expenses).")

    return advice


def full_analysis(income: float, total_expenses: float,
                  savings_goal: float, lifestyle_score: float,
                  predicted_savings: float) -> dict:
    return {
        "health_score": compute_health_score(income, predicted_savings),
        "overspending": detect_overspending(income, total_expenses),
        "advice": generate_advice(income, total_expenses, savings_goal, lifestyle_score, predicted_savings),
    }
