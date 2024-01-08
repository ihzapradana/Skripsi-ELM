import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import pinv
# from sklearn.metrics import mean_squared_error
# from math import sqrt

train = pd.read_csv('Training_70_f4_keju.csv')
train.head(174)

test = pd.read_csv('Testing_30_f4_keju.csv')
test.head(75)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train.values[:,1:5])
y_train = scaler.fit_transform(train.values[:,5:])

X_test = scaler.fit_transform(test.values[:,1:5])
y_test = scaler.fit_transform(test.values[:,5:])

input_size = X_train.shape[1]

hidden_size = 5

input_weights = np.random.uniform(low=-1, high=1, size=[input_size,hidden_size])
print("input_weights = ", input_weights)
biases = np.random.uniform(low=-1, high=1, size=[hidden_size])
print("Bias = ", biases)

# def sig(x):
#  return 1/(1 + np.exp(-x))

def bipolar_sigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

# def hidden_nodes(X):
#     G = np.dot(X, input_weights)
#     G = G + biases
#     H = sig(G)
#     return H

def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = bipolar_sigmoid(G)
    return H

output_weights = np.dot(pinv(hidden_nodes(X_train)), y_train)
print(output_weights)

def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

prediction = predict(X_test)
print("Prediction = ", prediction)

prediksi = scaler.inverse_transform(prediction)
print("Prediksi = ",prediksi)

y_true = np.array(test["X1"])
y_pred = np.array(prediksi)

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

print("MAPE = ", mean_absolute_percentage_error(y_true, y_pred), "%")