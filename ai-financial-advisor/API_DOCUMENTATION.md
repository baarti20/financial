# FastAPI Prediction Endpoint Documentation

## Overview
The AI Financial Advisor provides a FastAPI-based REST API for financial prediction and analysis. The primary prediction endpoint calculates estimated savings based on user financial inputs.

---

## Prediction Endpoint

### Endpoint URL
```
POST /predict
```

### Description
Predicts savings using the formula: **Income − (Fixed Expenses + Variable Expenses)**

Uses a trained Scikit-learn RandomForest model if available, otherwise falls back to rule-based calculation.

---

## Request Schema

### Content-Type
```
application/json
```

### Request Body - `FinancialInput`

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `income` | float | ✅ Yes | > 0 | Annual income in INR |
| `fixed_expenses` | float | ✅ Yes | ≥ 0 | Monthly fixed expenses (rent, fuel, insurance, utilities) summed for a month in INR |
| `variable_expenses` | float | ✅ Yes | ≥ 0 | Monthly variable expenses (food, transport, entertainment, shopping, medical, etc.) summed for a month in INR |
| `savings_goal` | float | ✅ Yes | ≥ 0 | Target annual savings goal in INR |
| `lifestyle_score` | float | ✅ Yes | 1 ≤ value ≤ 10 | Lifestyle preference score (1 = Frugal, 10 = Lavish) |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "income": 800000,
    "fixed_expenses": 18000,
    "variable_expenses": 11500,
    "savings_goal": 150000,
    "lifestyle_score": 5.5
  }'
```

### JavaScript Example (Fetch API)

```javascript
const response = await fetch('/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    income: 800000,
    fixed_expenses: 18000,
    variable_expenses: 11500,
    savings_goal: 150000,
    lifestyle_score: 5.5
  })
});

const data = await response.json();
console.log(data);
```

### Python Example (Requests)

```python
import requests

url = "http://127.0.0.1:8001/predict"
payload = {
    "income": 800000,
    "fixed_expenses": 18000,
    "variable_expenses": 11500,
    "savings_goal": 150000,
    "lifestyle_score": 5.5
}

response = requests.post(url, json=payload)
data = response.json()
print(data)
```

---

## Response Schema

### Success Response (200 OK)

```json
{
  "predicted_savings": 756500.00,
  "actual_savings_estimate": 770500.00
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted_savings` | float | ML model predicted annual savings (rounded to 2 decimals) |
| `actual_savings_estimate` | float | Rule-based savings calculation: Income − Total Monthly Expenses × 12 |

### Error Response (400 Bad Request)

```json
{
  "detail": [
    {
      "loc": ["body", "income"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt",
      "ctx": {"limit_value": 0}
    }
  ]
}
```

Validation errors occur when:
- `income` ≤ 0
- `lifestyle_score` < 1 or > 10
- Any numeric field is not a valid number
- Required fields are missing

---

## Implementation Details

### Model Loading
```python
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
```

### Prediction Function
```python
def _predict(data: FinancialInput) -> float:
    """
    Calculate predicted savings using ML model or fallback rule-based method.
    
    Args:
        data (FinancialInput): User financial input containing income, expenses, 
                               savings goal, and lifestyle score.
    
    Returns:
        float: Predicted annual savings rounded to 2 decimal places.
    """
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
    
    # Fallback: Simple rule-based calculation
    return round(data.income - total, 2)
```

### Endpoint Handler
```python
@app.post("/predict")
async def predict(data: FinancialInput):
    """Predict savings using: Income − (Fixed Expenses + Variable Expenses)."""
    predicted_savings = _predict(data)
    return {
        "predicted_savings": predicted_savings,
        "actual_savings_estimate": round(data.income - data.total_expenses, 2),
    }
```

---

## Calculation Example

### Input
```json
{
  "income": 800000,
  "fixed_expenses": 18000,
  "variable_expenses": 11500,
  "savings_goal": 150000,
  "lifestyle_score": 5.5
}
```

### Calculation Steps

1. **Monthly Total Expenses:**
   - Fixed: ₹18,000
   - Variable: ₹11,500
   - Monthly Total: ₹29,500

2. **Annual Total Expenses:**
   - ₹29,500 × 12 = ₹354,000

3. **Rule-Based Savings (Fallback):**
   - ₹800,000 − ₹354,000 = ₹446,000

4. **ML Model Prediction (If Available):**
   - Input Features: [800000, 29500, 150000, 5.5]
   - RandomForest Model Output: ₹756,500 (example)

### Output
```json
{
  "predicted_savings": 756500.00,
  "actual_savings_estimate": 446000.00
}
```

---

## Error Handling

### Validation Errors

The endpoint validates all input fields:

```python
class FinancialInput(BaseModel):
    income: float = Field(..., gt=0, description="Annual income in INR")
    fixed_expenses: float = Field(..., ge=0)
    variable_expenses: float = Field(..., ge=0)
    savings_goal: float = Field(..., ge=0)
    lifestyle_score: float = Field(..., ge=1, le=10)
```

**Common Error Cases:**

1. **Income ≤ 0**
   ```json
   {
     "detail": [
       {
         "loc": ["body", "income"],
         "msg": "ensure this value is greater than 0",
         "type": "value_error.number.not_gt"
       }
     ]
   }
   ```

2. **Lifestyle Score Out of Range**
   ```json
   {
     "detail": [
       {
         "loc": ["body", "lifestyle_score"],
         "msg": "ensure this value is less than or equal to 10",
         "type": "value_error.number.not_le"
       }
     ]
   }
   ```

3. **Missing Required Field**
   ```json
   {
     "detail": [
       {
         "loc": ["body", "savings_goal"],
         "msg": "field required",
         "type": "value_error.missing"
       }
     ]
   }
   ```

---

## HTTP Status Codes

| Status Code | Meaning | Example |
|-------------|---------|---------|
| `200 OK` | Prediction successful | `{"predicted_savings": 756500.00, "actual_savings_estimate": 446000.00}` |
| `400 Bad Request` | Invalid input data (validation error) | Missing field or invalid value |
| `422 Unprocessable Entity` | Request body parsing error | Invalid JSON format |
| `500 Internal Server Error` | Server error during processing | Unexpected exception |

---

## Usage Workflow

### Step 1: Prepare User Input
Collect financial data from user form:
- Annual income
- Monthly fixed expenses breakdown
- Monthly variable expenses breakdown
- Annual savings goal
- Lifestyle score (1-10 slider)

### Step 2: Call Prediction Endpoint
```javascript
const response = await fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    income: userInputForm.income,
    fixed_expenses: userInputForm.fixedExpenses,
    variable_expenses: userInputForm.variableExpenses,
    savings_goal: userInputForm.savingsGoal,
    lifestyle_score: userInputForm.lifestyleScore
  })
});
```

### Step 3: Process Response
```javascript
const data = await response.json();

if (response.ok) {
  console.log("Predicted Savings:", data.predicted_savings);
  console.log("Actual Estimate:", data.actual_savings_estimate);
  // Display results to user
} else {
  console.error("Validation Error:", data.detail);
  // Show error messages to user
}
```

---

## Model Details

### Training Process
- **Dataset:** Synthetic financial data (`ml/financial_data.csv`)
- **Algorithm:** Scikit-learn RandomForest Regressor
- **Features:** Income, Total Expenses, Savings Goal, Lifestyle Score
- **Target:** Predicted Annual Savings
- **Model File:** `ml/model.pkl`

### Fallback Mechanism
If the trained model is unavailable or fails to load:
1. Endpoint attempts to retrain model from dataset
2. If retraining fails, uses rule-based calculation: `Income − Total Expenses`
3. No errors are raised; API continues functioning with fallback logic

---

## API Specifications

### Content Negotiation
- **Request:** `application/json` (required)
- **Response:** `application/json`

### Async Support
The endpoint is async-compatible:
```python
@app.post("/predict")
async def predict(data: FinancialInput):
    ...
```

### CORS Enabled
All endpoints allow cross-origin requests:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Related Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Predict savings only |
| `/analyze` | POST | Get health score + overspending analysis |
| `/advisor` | POST | Full analysis + personalized advice (requires auth) |
| `/history` | GET | View past analysis records (requires auth) |

---

## Version Information

- **API Version:** 1.0.0
- **Framework:** FastAPI 0.111.0+
- **Python:** 3.10+
- **Dependencies:** pandas, scikit-learn, numpy, scipy

---

## Support & Troubleshooting

### Issue: "Model failed to load"
- Ensure `ml/model.pkl` exists
- Check file permissions
- Verify dataset at `ml/financial_data.csv` is present
- API will fallback to rule-based calculation automatically

### Issue: Validation errors on valid input
- Ensure all numeric values are proper numbers (not strings)
- Check `lifestyle_score` is between 1-10
- Verify `income` is greater than 0
- Confirm all required fields are included

### Issue: Incorrect predictions
- If using fallback mode: predictions are simple Income − Expenses calculation
- For ML predictions: model accuracy depends on training dataset quality
- Consider retraining model with fresh data if predictions seem off

---

## Performance Notes

- **Average Response Time:** < 100ms
- **Model Prediction Time:** ~5-10ms (negligible compared to HTTP overhead)
- **Fallback Calculation Time:** < 1ms
- **No Database Calls:** `/predict` endpoint is stateless and fast

---

## Security Considerations

- The `/predict` endpoint is **public** (no authentication required)
- Use `/advisor` endpoint for authenticated predictions with record storage
- Set appropriate rate limiting in production
- CORS is unrestricted in development; configure appropriately for production
- Input validation prevents injection attacks

---

## Future Enhancements

- [ ] Add request rate limiting
- [ ] Implement response caching
- [ ] Add confidence intervals to predictions
- [ ] Support for multiple currency formats
- [ ] Time-series predictions (multi-month)
- [ ] Portfolio optimization suggestions

