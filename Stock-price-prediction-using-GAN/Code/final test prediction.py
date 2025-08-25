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
    default=3,
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

# Add x_train_path and y_train_path for training evaluation
x_train_path = "X_train.npy"
y_train_path = "y_train.npy"
index_train_path = "index_train.npy"

# --- Check for Required Files ---
required_files = [model_path, x_test_path, y_test_path, index_test_path, y_scaler_path,
                  x_train_path, y_train_path, index_train_path] # Added train files
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
print("Loading scaler and data (Train & Test)...")
y_scaler = load(open(y_scaler_path, 'rb'))
X_test = np.load(x_test_path, allow_pickle=True)
y_test_scaled = np.load(y_test_path, allow_pickle=True) # Scaled actual target values
test_predict_index = np.load(index_test_path, allow_pickle=True) # Timestamps for test predictions

# Load training data for evaluation
X_train = np.load(x_train_path, allow_pickle=True)
y_train_scaled = np.load(y_train_path, allow_pickle=True)
train_predict_index = np.load(index_train_path, allow_pickle=True)


print("Shapes:")
print(f"  X_train: {X_train.shape}, y_train_scaled: {y_train_scaled.shape}, train_index: {train_predict_index.shape}")
print(f"  X_test: {X_test.shape}, y_test_scaled: {y_test_scaled.shape}, test_index: {test_predict_index.shape}")


# --- Validate Data Shape (Basic Checks) ---
n_steps_out = 1 # For a 3-to-1 model
if X_test.shape[1] != N_STEPS_IN or y_test_scaled.shape[1] != n_steps_out:
    print(f"Error: Test Data shape mismatch. Expected X_test steps={N_STEPS_IN}, y_test_scaled steps={n_steps_out}.")
    sys.exit(1)
if X_train.shape[1] != N_STEPS_IN or y_train_scaled.shape[1] != n_steps_out:
    print(f"Error: Train Data shape mismatch. Expected X_train steps={N_STEPS_IN}, y_train_scaled steps={n_steps_out}.")
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
# == Step 1: Evaluate on Train Set, Full Test Set, and Subsets ==
# ==============================================================
print("\n--- Step 1: Evaluating Model Performance ---")

# --- 1.a) Evaluate on TRAINING SET ---
print("Making predictions on the training set...")
y_pred_train_scaled = model.predict(X_train)

# Reshape/Validate Train Predictions
if len(y_pred_train_scaled.shape) == 3 and y_pred_train_scaled.shape[2] == 1:
    y_pred_train_scaled = np.squeeze(y_pred_train_scaled, axis=-1)
if len(y_pred_train_scaled.shape) == 1:
     y_pred_train_scaled = y_pred_train_scaled.reshape(-1, 1)
if y_pred_train_scaled.shape != y_train_scaled.shape:
    print(f"Error: Train prediction shape {y_pred_train_scaled.shape} mismatch target {y_train_scaled.shape}.")
    sys.exit(1)

# Inverse Scale Train Results
rescaled_train_predicted_y = y_scaler.inverse_transform(y_pred_train_scaled)
rescaled_train_real_y = y_scaler.inverse_transform(y_train_scaled)

# Create Train Results DataFrame
train_results_df = pd.DataFrame({
    'Real_Price': rescaled_train_real_y.flatten(),
    'Predicted_Price': rescaled_train_predicted_y.flatten()
}, index=pd.to_datetime(train_predict_index)) # Use train index

# Calculate Train RMSE
train_rmse = np.sqrt(mean_squared_error(train_results_df['Real_Price'], train_results_df['Predicted_Price']))
print(f"  Train Set RMSE: {train_rmse:.4f}")


# --- 1.b) Evaluate on FULL TEST SET ---
print("\nMaking predictions on the full test set...")
y_pred_test_scaled = model.predict(X_test)

# Reshape/Validate Test Predictions
if len(y_pred_test_scaled.shape) == 3 and y_pred_test_scaled.shape[2] == 1:
    y_pred_test_scaled = np.squeeze(y_pred_test_scaled, axis=-1)
if len(y_pred_test_scaled.shape) == 1:
     y_pred_test_scaled = y_pred_test_scaled.reshape(-1, 1)
if y_pred_test_scaled.shape != y_test_scaled.shape:
    print(f"Error: Test prediction shape {y_pred_test_scaled.shape} mismatch target {y_test_scaled.shape}.")
    sys.exit(1)

# Inverse Scale Test Results
rescaled_test_predicted_y = y_scaler.inverse_transform(y_pred_test_scaled)
rescaled_test_real_y = y_scaler.inverse_transform(y_test_scaled)

# Create Test Results DataFrame
test_results_df = pd.DataFrame({
    'Real_Price': rescaled_test_real_y.flatten(),
    'Predicted_Price': rescaled_test_predicted_y.flatten()
}, index=pd.to_datetime(test_predict_index)) # Use test index

# Calculate Overall Test RMSE
test_rmse_overall = np.sqrt(mean_squared_error(test_results_df['Real_Price'], test_results_df['Predicted_Price']))
print(f"  Overall Test Set RMSE: {test_rmse_overall:.4f}")


# --- 1.c) Evaluate on TEST SET WITHOUT 2020 ---
print("\nFiltering test set results (excluding year 2020)...")
test_results_without_2020 = test_results_df[test_results_df.index.year != 2020]

if test_results_without_2020.empty:
    print("  No test data found excluding year 2020.")
    test_rmse_without_2020 = np.nan
else:
    test_rmse_without_2020 = np.sqrt(mean_squared_error(
        test_results_without_2020['Real_Price'],
        test_results_without_2020['Predicted_Price']
    ))
    print(f"  Test Set (excluding 2020) RMSE: {test_rmse_without_2020:.4f}")


# --- 1.d) Evaluate on TEST SET ONLY 2020 ---
print("\nFiltering test set results (only year 2020)...")
test_results_only_2020 = test_results_df[test_results_df.index.year == 2020]

if test_results_only_2020.empty:
    print("  No test data found for year 2020.")
    test_rmse_only_2020 = np.nan
else:
    test_rmse_only_2020 = np.sqrt(mean_squared_error(
        test_results_only_2020['Real_Price'],
        test_results_only_2020['Predicted_Price']
    ))
    print(f"  Test Set (only 2020) RMSE: {test_rmse_only_2020:.4f}")

# --- Summary of Evaluation RMSEs ---
print("\n--- Evaluation RMSE Summary ---")
print(f"  Training Set:      {train_rmse:.4f}")
print(f"  Overall Test Set:  {test_rmse_overall:.4f}")
print(f"  Test (Excl. 2020): {test_rmse_without_2020:.4f}")
print(f"  Test (Only 2020):  {test_rmse_only_2020:.4f}")
print("----------------------------")


# --- Plot Overall Test Set Results ---
print("\nGenerating overall test set plot...")
plt.figure(figsize=(15, 7))
plt.plot(test_results_df.index, test_results_df["Real_Price"], label='Actual Price', color='blue', linewidth=1.5)
plt.plot(test_results_df.index, test_results_df["Predicted_Price"], label='Predicted Price (Test Set)', color='orange', linestyle='--', linewidth=1.2)
plt.title(f'Overall Test Set: Actual vs. Predicted ({model_filename})\nRMSE: {test_rmse_overall:.4f}', fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Stock Price (USD)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

# --- Save Overall Test Set Plot and Results ---
base_filename = os.path.splitext(model_filename)[0]
test_plot_filename = f"overall_test_set_plot_{base_filename}.png"
test_results_filename = f"overall_test_set_results_{base_filename}.csv"

print(f"Saving overall test set plot to '{test_plot_filename}'...")
plt.savefig(test_plot_filename)
print(f"Saving overall test set results to '{test_results_filename}'...")
# Save the full test results including all columns if needed
test_results_df.to_csv(test_results_filename)
plt.show() # Display the overall test set plot

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
    # Reshape current_sequence for prediction: (1, N_STEPS_IN, num_features)
    input_for_prediction = current_sequence[np.newaxis, :, :]

    # Predict the next step (output shape depends on model, assume (1, 1))
    next_scaled_close_pred_arr = model.predict(input_for_prediction)

    # Extract the scalar prediction value
    # Handle potential shape variations (e.g., (1,1) vs (1,) )
    if isinstance(next_scaled_close_pred_arr, np.ndarray) and next_scaled_close_pred_arr.ndim >= 1:
         next_scaled_close_pred = next_scaled_close_pred_arr.item(0) # Get first element
    else:
         next_scaled_close_pred = next_scaled_close_pred_arr # Assume it's already scalar

    predicted_scaled_closes.append(next_scaled_close_pred)

    # --- Update the sequence for the next prediction ---
    # Create the next feature vector (copy the last known features)
    next_feature_vector = current_sequence[-1, :].copy()
    # Update the target feature (Close price) with the prediction
    next_feature_vector[TARGET_COL_INDEX] = next_scaled_close_pred

    # Roll the sequence (discard oldest step)
    current_sequence = np.roll(current_sequence, shift=-1, axis=0)
    # Append the new feature vector as the last step
    current_sequence[-1, :] = next_feature_vector

    # Optional: print(f"  Forecasted Day {i+1}...")

print("Forecasting loop completed.")

# --- Process Forecast Results ---
predicted_scaled_closes = np.array(predicted_scaled_closes).reshape(-1, 1)
predicted_prices = y_scaler.inverse_transform(predicted_scaled_closes)

# Determine forecast dates (using Business Day frequency)
last_known_date = pd.to_datetime(test_predict_index[-1])
forecast_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1),
                               periods=days_to_predict,
                               freq='B') # Use Business Day frequency

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