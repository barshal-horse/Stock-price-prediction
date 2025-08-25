# forecast_future_with_test_eval.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load
from sklearn.metrics import mean_squared_error
import argparse
import sys # To exit gracefully

# --- Dummy Function for WGAN-GP Loading (if needed) ---
def wasserstein_loss(y_true, y_pred):
    """ Placeholder Wasserstein loss for loading WGAN-GP models. """
    return tf.reduce_mean(y_true * y_pred)

# --- Constants ---
MODEL_DIR = "Models"
# !!! CRITICAL ASSUMPTION: Based on your 'data_preprocessing.py', the 'Close' price
#     was the 4th column (index 3) BEFORE 'Date' was dropped and set as index.
#     Verify this index if your feature set changes.
TARGET_COL_INDEX = 3
N_STEPS_IN = 3 # This script assumes the model uses 3 input steps

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description=f"Evaluate a {N_STEPS_IN}-to-1 model on the test set and forecast future stock prices."
)
parser.add_argument(
    "model_filename",
    help=f"Filename of the {N_STEPS_IN}-to-1 model in '{MODEL_DIR}'. E.g., 'WGAN_GP_3to1.h5'"
)
parser.add_argument(
    "--days",
    type=int,
    default=10,
    help="Number of future days to forecast (after test set evaluation)."
)
args = parser.parse_args()
model_filename = args.model_filename
days_to_predict = args.days

# --- Validation ---
if f"_{N_STEPS_IN}to1.h5" not in model_filename:
    print(f"Warning: Expected model filename ending in '_{N_STEPS_IN}to1.h5'.")
    # Allow proceeding but warn the user model might be incompatible.

# --- Define File Paths ---
model_path = os.path.join(MODEL_DIR, model_filename)
x_test_path = "X_test.npy"
y_test_path = "y_test.npy" # Need this for actual values on test set
index_test_path = "index_test.npy"
y_scaler_path = "y_scaler.pkl"

# --- Check for Required Files ---
# Add y_test_path to required files
required_files = [model_path, x_test_path, y_test_path, index_test_path, y_scaler_path]
print("Checking for required files...")
all_files_found = True
for f_path in required_files:
    if not os.path.exists(f_path):
        print(f"- Missing file: {f_path}")
        all_files_found = False

if not all_files_found:
    print("\nError: One or more required files are missing.")
    print("Please ensure:")
    print("  1. You have run 'Load_data.py' and 'data_preprocessing.py' successfully.")
    print(f"  2. The model file '{model_filename}' exists in the '{MODEL_DIR}' directory.")
    sys.exit(1)
print("All required files found.")

print(f"\n--- Evaluating & Forecasting using: {model_filename} ---")

# --- Load Scaler and Data ---
print("Loading scaler and test data...")
y_scaler = load(open(y_scaler_path, 'rb'))
X_test = np.load(x_test_path, allow_pickle=True)
y_test_scaled = np.load(y_test_path, allow_pickle=True) # Scaled actual target values
test_predict_index = np.load(index_test_path, allow_pickle=True) # Timestamps for test predictions

print(f"  X_test shape: {X_test.shape}")
print(f"  y_test_scaled shape: {y_test_scaled.shape}")
print(f"  Test index shape: {test_predict_index.shape}")

# --- Validate Test Data Shape ---
n_steps_out = 1 # For a 3-to-1 model
if X_test.shape[1] != N_STEPS_IN or y_test_scaled.shape[1] != n_steps_out:
    print(f"Error: Data shape mismatch. Expected X_test steps={N_STEPS_IN}, y_test_scaled steps={n_steps_out}.")
    sys.exit(1)
if len(X_test) != len(y_test_scaled) or len(X_test) != len(test_predict_index):
     print("Error: Mismatch in number of samples between X_test, y_test_scaled, and index_test.")
     sys.exit(1)

# --- Load Keras Model ---
print(f"Loading model: {model_path}...")
custom_objects = {}
if 'WGAN_GP' in model_filename:
    print("  (WGAN-GP model detected, adding 'wasserstein_loss' custom object)")
    custom_objects['wasserstein_loss'] = wasserstein_loss

try:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    model.summary()
except Exception as e:
    print(f"\nError loading Keras model: {e}")
    sys.exit(1)

# ==============================================================
# == Step 1: Evaluate on the Existing Test Set ==
# ==============================================================
print("\n--- Step 1: Evaluating on Test Set ---")
print("Making predictions on the test set...")
y_pred_test_scaled = model.predict(X_test)

# --- Reshape Test Predictions if Necessary ---
if len(y_pred_test_scaled.shape) == 3 and y_pred_test_scaled.shape[2] == 1:
    y_pred_test_scaled = np.squeeze(y_pred_test_scaled, axis=-1)
if len(y_pred_test_scaled.shape) == 1:
     y_pred_test_scaled = y_pred_test_scaled.reshape(-1, 1)
if y_pred_test_scaled.shape != y_test_scaled.shape:
    print(f"Error: Test prediction shape {y_pred_test_scaled.shape} mismatch target {y_test_scaled.shape}.")
    sys.exit(1)
print(f"  Test prediction shape: {y_pred_test_scaled.shape}")

# --- Inverse Scale Test Results ---
print("Inverse scaling test set results...")
rescaled_test_predicted_y = y_scaler.inverse_transform(y_pred_test_scaled)
rescaled_test_real_y = y_scaler.inverse_transform(y_test_scaled)

# --- Create Test Results DataFrame ---
test_results_df = pd.DataFrame({
    'Real_Price': rescaled_test_real_y.flatten(),
    'Predicted_Price': rescaled_test_predicted_y.flatten()
}, index=pd.to_datetime(test_predict_index))

# --- Calculate Test RMSE ---
test_rmse = np.sqrt(mean_squared_error(test_results_df['Real_Price'], test_results_df['Predicted_Price']))
print(f"  Test Set RMSE: {test_rmse:.4f}")

# --- Plot Test Set Results ---
print("Generating test set plot...")
plt.figure(figsize=(15, 7))
plt.plot(test_results_df.index, test_results_df["Real_Price"], label='Actual Price', color='blue', linewidth=1.5)
plt.plot(test_results_df.index, test_results_df["Predicted_Price"], label='Predicted Price (Test Set)', color='orange', linestyle='--', linewidth=1.2)
plt.title(f'Test Set: Actual vs. Predicted ({model_filename})\nRMSE: {test_rmse:.4f}', fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Stock Price (USD)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

# --- Save Test Set Plot and Results ---
base_filename = os.path.splitext(model_filename)[0]
test_plot_filename = f"test_set_plot_{base_filename}.png"
test_results_filename = f"test_set_results_{base_filename}.csv"

print(f"Saving test set plot to '{test_plot_filename}'...")
plt.savefig(test_plot_filename)
print(f"Saving test set results to '{test_results_filename}'...")
test_results_df.to_csv(test_results_filename)
plt.show() # Display the test set plot

# ==============================================================
# == Step 2: Forecast Future Prices (Iterative) ==
# ==============================================================
print(f"\n--- Step 2: Forecasting Future {days_to_predict} Days ---")

# Get the last known sequence of features (shape: (N_STEPS_IN, num_features))
last_sequence = X_test[-1, :, :].copy()
num_features = last_sequence.shape[1]
print(f"  Using last sequence from X_test with shape: {last_sequence.shape}")
if TARGET_COL_INDEX >= num_features:
    print(f"Error: TARGET_COL_INDEX ({TARGET_COL_INDEX}) is out of bounds.")
    sys.exit(1)

# --- Iterative Forecasting Loop ---
print(f"Starting iterative forecast...")
predicted_scaled_closes = []
current_sequence = last_sequence.copy()

for i in range(days_to_predict):
    input_for_prediction = current_sequence[np.newaxis, :, :]
    next_scaled_close_pred = model.predict(input_for_prediction)[0, 0]
    predicted_scaled_closes.append(next_scaled_close_pred)

    next_feature_vector = current_sequence[-1, :].copy()
    next_feature_vector[TARGET_COL_INDEX] = next_scaled_close_pred

    current_sequence = np.roll(current_sequence, shift=-1, axis=0)
    current_sequence[-1, :] = next_feature_vector
    # Optional: print(f"  Forecasted Day {i+1}...")

print("Forecasting loop completed.")

# --- Process Forecast Results ---
predicted_scaled_closes = np.array(predicted_scaled_closes).reshape(-1, 1)
predicted_prices = y_scaler.inverse_transform(predicted_scaled_closes)

last_known_date = pd.to_datetime(test_predict_index[-1])
forecast_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1),
                               periods=days_to_predict,
                               freq='B')

forecast_df = pd.DataFrame({
    'Predicted_Price': predicted_prices.flatten()
}, index=forecast_dates)
forecast_df.index.name = 'Forecast_Date'

print("\n--- Forecast Results ---")
print(forecast_df)

# --- Plot Forecast Results ---
print("Generating forecast plot...")
plt.figure(figsize=(12, 6))
plt.plot(forecast_df.index, forecast_df["Predicted_Price"], marker='o', linestyle='-', label='Forecasted Price')
plt.title(f'{days_to_predict}-Day Future Forecast using {model_filename}', fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Predicted Stock Price (USD)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

# --- Save Forecast Plot and Results ---
forecast_plot_filename = f"future_forecast_plot_{base_filename}_{days_to_predict}days.png"
forecast_results_filename = f"future_forecast_results_{base_filename}_{days_to_predict}days.csv"

print(f"Saving forecast plot to '{forecast_plot_filename}'...")
plt.savefig(forecast_plot_filename)
print(f"Saving forecast results to '{forecast_results_filename}'...")
forecast_df.to_csv(forecast_results_filename)
plt.show() # Display the forecast plot

print(f"\n--- Evaluation and Forecast Completed Successfully ---")