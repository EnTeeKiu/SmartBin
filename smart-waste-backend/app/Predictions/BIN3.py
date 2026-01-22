import pandas as pd
import joblib
from datetime import date, timedelta

MODEL_PATH = "smart-waste-backend/app/models/BIN3.pkl"

model = joblib.load(MODEL_PATH)

def predict_trash_generated_tomorrow(
    trash_today,
    trash_yesterday,
    trash_2_days_ago,
    tomorrow_date,
    is_holiday_tomorrow
):
    day_of_week = tomorrow_date.weekday()
    is_weekend = int(day_of_week >= 5)

    X = pd.DataFrame([{
        "trash_generated_today": trash_today,
        "trash_generated_yesterday": trash_yesterday,
        "trash_generated_2_days_ago": trash_2_days_ago,
        "is_weekend_tomorrow": is_weekend,
        "is_holiday_tomorrow": int(is_holiday_tomorrow),
    }])

    prediction = model.predict(X)[0]

    # Trash demand cannot be negative
    return max(0.0, float(prediction))
