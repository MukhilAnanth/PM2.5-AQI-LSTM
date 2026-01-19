import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib

TIME_STEPS = 365
EPOCHS = 25
BATCH_SIZE = 16

data = pd.read_csv("FilteredPM25AQI.csv")

print(data.columns)
PM25 = pd.to_numeric(data.iloc[1:2197, 2], errors='coerce').dropna().values
print("Number of valid samples:", len(PM25))



scaler = MinMaxScaler()
PM25_scaled = scaler.fit_transform(PM25.reshape(-1, 1))

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(PM25_scaled, TIME_STEPS)
X = X.reshape(X.shape[0], TIME_STEPS, 1)

model = keras.Sequential([
    layers.Input(shape=(TIME_STEPS, 1)),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

callback = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

model.fit(X,y,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.2,shuffle=False,callbacks=[callback])

model.save("pm25_lstm_model.keras")
joblib.dump(scaler, "pm25_scaler.pkl")

print("Trained and Completed")
