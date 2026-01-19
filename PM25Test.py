import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta

TIME_STEPS = 365
START_DATE = "2024-12-01"
END_DATE   = "2024-12-31"


model = load_model("pm25_lstm_model.keras")
scaler = joblib.load("pm25_scaler.pkl")

data = pd.read_csv("FilteredPM25AQI.csv")

data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], errors="coerce")
data = data.dropna(subset=[data.columns[0]])

data = data.sort_values(by=data.columns[0])


start_date = pd.to_datetime(START_DATE)
end_date   = pd.to_datetime(END_DATE)

current_date = start_date
predictions = []
actuals = []

while current_date <= end_date:

    previous_day = current_date - timedelta(days=1)

    history = data[data.iloc[:, 0] <= previous_day]

    PM25 = pd.to_numeric(history.iloc[:, 2], errors="coerce").dropna().values

    if len(PM25) < TIME_STEPS:
        current_date += timedelta(days=1)
        continue

    PM25_scaled = scaler.transform(PM25.reshape(-1, 1))

    last_sequence = PM25_scaled[-TIME_STEPS:].reshape(1, TIME_STEPS, 1)

    pred_scaled = model.predict(last_sequence, verbose=0)
    pred_pm25 = scaler.inverse_transform(pred_scaled)[0][0]

    actual_row = data[data.iloc[:, 0] == current_date]
    if not actual_row.empty:
        actual_pm25 = pd.to_numeric(
            actual_row.iloc[0, 2], errors="coerce"
        )

        if not np.isnan(actual_pm25):
            predictions.append(pred_pm25)
            actuals.append(actual_pm25)

            print(
                f"{current_date.date()} | "
                f"Pred: {pred_pm25:.2f} | "
                f"Actual: {actual_pm25:.2f}"
            )

    current_date += timedelta(days=1)


predictions = np.array(predictions)
actuals = np.array(actuals)

mae = np.mean(np.abs(predictions - actuals))
rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

print("\nDecember 2024 Results")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
