import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pickle import load

# Load models
model_30to3 = tf.keras.models.load_model("Models/WGAN_GP_30to3.h5")
model_3to1 = tf.keras.models.load_model("Models/WGAN_GP_3to1.h5")

# Load scalers
X_scaler = load(open("X_scaler.pkl", "rb"))
y_scaler = load(open("y_scaler.pkl", "rb"))

# Load test dataset
X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)
test_predict_index = np.load("index_test.npy", allow_pickle=True)

# Ensure X_test has the correct shape
if X_test.shape[1] == 3:  # If data has 3 timesteps instead of 30
    print(f"Reshaping X_test from {X_test.shape} to (samples, 30, features)...")
    X_test_padded = np.pad(X_test, ((0, 0), (27, 0), (0, 0)), mode='edge')
else:
    X_test_padded = X_test

num_features = X_test_padded.shape[2]  # Extract features dynamically

def predict_stock_prices(X_test, model_30to3, model_3to1, days_to_predict):
    predictions = []
    
    last_3_days = X_test[-1, -3:, :].copy()  # Extract last 3 timesteps

    for _ in range(days_to_predict):
        if last_3_days.shape != (3, 36):  # Ensure correct shape
            raise ValueError(f"Expected shape (3, 36), but got {last_3_days.shape}")

        next_day_pred = model_3to1.predict(last_3_days[np.newaxis, :, :])[0]  # Ensure batch dim
        
        predictions.append(next_day_pred)
        last_3_days = np.roll(last_3_days, shift=-1, axis=0)  # Shift data
        last_3_days[-1, :] = next_day_pred  # Update last timestep with prediction

    return np.vstack(predictions)  # Ensure correct shape

# Run predictions
predicted_results = predict_stock_prices(X_test_padded, model_30to3, model_3to1, days_to_predict=10)

# Plot results
plt.figure(figsize=(16, 8))
plt.plot(predicted_results[:, 0], label="Predicted Price", color='r')  # Assuming first column is the stock price
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Stock Price Forecast")
plt.savefig("stock_prediction.png")
plt.show()

# Save predictions
df_predictions = pd.DataFrame(predicted_results, columns=[f"Feature_{i+1}" for i in range(predicted_results.shape[1])])
df_predictions.to_csv("predicted_stock_prices.csv", index=False)

print("Predictions saved successfully!")
