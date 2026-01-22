import pandas as pd
import numpy as np

# ================= CONFIG =================
INPUT_CSV = "smart-waste-backend/data/BIN5.csv"
OUTPUT_CSV = "smart-waste-backend/data/BIN5-Processed.csv"

# ================= LOAD =================
df = pd.read_csv(INPUT_CSV)
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(by=["bin_id", "date"]).reset_index(drop=True)

# ================= PREP =================
df["fullness_yesterday"] = (
    df.groupby("bin_id")["fullness_today"].shift(1)
)

df["pickup_yesterday"] = (
    df.groupby("bin_id")["pickup_today"].shift(1)
)

df["fullness_tomorrow"] = (
    df.groupby("bin_id")["fullness_today"].shift(-1)
)

df["pickup_tomorrow"] = (
    df.groupby("bin_id")["pickup_today"].shift(-1)
)

# ================= TRASH GENERATED TODAY =================
df["trash_generated_today"] = (
    df["fullness_today"] - df["fullness_yesterday"]
)

# If bin was emptied yesterday → reset
mask_reset_today = df["pickup_yesterday"] == 1
df.loc[mask_reset_today, "trash_generated_today"] = (
    df.loc[mask_reset_today, "fullness_today"]
)

# ================= LAGS =================
df["trash_generated_yesterday"] = (
    df.groupby("bin_id")["trash_generated_today"].shift(1)
)

df["trash_generated_2_days_ago"] = (
    df.groupby("bin_id")["trash_generated_today"].shift(2)
)

# ================= TARGET: TOMORROW =================
df["target_trash_generated_tomorrow"] = (
    df["fullness_tomorrow"] - df["fullness_today"]
)

# If pickup today → tomorrow starts from 0
mask_reset_tomorrow = df["pickup_today"] == 1
df.loc[mask_reset_tomorrow, "target_trash_generated_tomorrow"] = (
    df.loc[mask_reset_tomorrow, "fullness_tomorrow"]
)

# ================= CALENDAR (TOMORROW) =================
df["date_tomorrow"] = df["date"] + pd.Timedelta(days=1)
df["day_of_week_tomorrow"] = df["date_tomorrow"].dt.weekday
df["is_weekend_tomorrow"] = (df["day_of_week_tomorrow"] >= 5).astype(int)

df["is_holiday_tomorrow"] = (
    df.groupby("bin_id")["is_holiday_today"].shift(-1)
)

# ================= FINAL DATASET =================
df_final = df.dropna().reset_index(drop=True)

training_df = df_final[
    [
        "trash_generated_today",
        "trash_generated_yesterday",
        "trash_generated_2_days_ago",
        "day_of_week_tomorrow",
        "is_weekend_tomorrow",
        "is_holiday_tomorrow",
        "target_trash_generated_tomorrow",
    ]
]

training_df.to_csv(OUTPUT_CSV, index=False)

print("✅ Feature engineering completed (correct timeline)")
print("Rows:", len(training_df))
print("Min values:")
print(training_df.min())
