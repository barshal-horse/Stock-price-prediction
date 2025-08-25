import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load
from sklearn.metrics import mean_squared_error

# Load scaler/index
X_scaler = load(open('X_scaler.pkl', 'rb'))
y_scaler = load(open('y_scaler.pkl', 'rb'))
train_predict_index = np.load("index_train.npy", allow_pickle=True)
test_predict_index = np.load("index_test.npy", allow_pickle=True)

# Load test dataset/model
G_model = tf.keras.models.load_model('gen_GRU_model_89.h5')
X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)

def predict_stock_prices(X_test, G_model):
    y_predicted = G_model.predict(X_test)
    rescaled_predicted_y = y_scaler.inverse_transform(y_predicted)
    return rescaled_predicted_y

def get_test_plot(X_test, y_test):
    output_dim = y_test.shape[1]
    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = predict_stock_prices(X_test, G_model)

    # Predicted price
    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["predicted_price"],
                                 index=test_predict_index[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
    
    # Real price
    real_price = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["real_price"],
                                index=test_predict_index[i:i + output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)
    
    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)
    
    # Plot results
    plt.figure(figsize=(16, 8))
    plt.plot(real_price["real_mean"], label="Real price")
    plt.plot(predict_result["predicted_mean"], color='r', label="Predicted price")
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(loc="upper left", fontsize=16)
    plt.title("The result of test", fontsize=20)
    plt.savefig('test_plot.png')
    plt.show()
    
    # Calculate RMSE
    RMSE = np.sqrt(mean_squared_error(predict_result["predicted_mean"], real_price["real_mean"]))
    print('-- RMSE -- ', RMSE)
    
    return predict_result, RMSE

test_predicted, test_RMSE = get_test_plot(X_test, y_test)
test_predicted.to_csv("test_predicted.csv")