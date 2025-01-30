# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 22:05:38 2025

@author: Jonathan Gonzalez

Machine Learning Regression Masterclass in Python 
By: Dr. Ryan Ahmed 
Platform: Udemy
Type: Compilation of videos

A neural network model is built using Keras, with multiple hidden layers and the ReLU 
activation function, trained using the Adam optimizer. The model's performance is 
evaluated using metrics like RMSE, MAE, and R², and predictions are compared against 
actual values. Visualizations, including scatter plots and training loss graphs, help 
interpret the results, making the model effective for house price prediction.

Last Updated: 1/29/2024
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

HouseData = pd.read_csv("kc_house_data.csv", encoding = "ISO-8859-1")
print(HouseData.head(5))
print(HouseData.tail(10))

HouseData.info()

print(HouseData.describe())

plt.close("all")

sns.scatterplot(x = "sqft_living", y = "price", data = HouseData)

HouseData.hist(bins = 20, figsize = (15,15), color = "r")

f, ax = plt.subplots(figsize = (10,10))
sns.heatmap((HouseData.drop(["id", "date"], axis = 1)).corr(), annot = True)

sns.pairplot(HouseData)

HouseDataSample = HouseData[ ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement", "yr_built" ] ]

sns.pairplot(HouseDataSample)

selected_features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above", "sqft_basement"]

x = HouseData[selected_features]

y = HouseData["price"]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)

print(scaler.data_max_)

print(scaler.data_min_)

y = y.values.reshape(-1,1)

y_scaled = scaler.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
y_test_scaled = scaler.fit_transform(y_test)
y_train_scaled = scaler.fit_transform(y_train)

import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim = 7, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(1, activation = "linear"))

print(model.summary())

model.compile(optimizer = "Adam", loss = "mean_squared_error")

epochs_hist = model.fit(x_train_scaled, y_train_scaled, epochs = 100, batch_size = 50, validation_split = 0.2)

epochs_hist.history.keys()

plt.figure()
plt.plot(epochs_hist.history["loss"])
plt.plot(epochs_hist.history["val_loss"])
plt.title("Model Loss Profess During Training")
plt.xlabel("Epoch")
plt.ylabel("Training and Validation Loss")
plt.legend(["Training Loss", "Validation Loss"])

x_test_1 = np.array([[ 4, 3, 1960, 5000, 1, 2000, 3000 ]])

scaler_1 = MinMaxScaler()

x_test_scaled_1 = scaler_1.fit_transform(x_test_1)

y_predict_1 = model.predict(x_test_scaled_1)

y_predict_1 = scaler.inverse_transform(y_predict_1)

y_predict = model.predict(x_test_scaled)

plt.figure()
plt.plot(y_test, y_predict, "^", color = "r")
plt.ylabel("Model Predictions")
plt.xlabel("True Values")

y_predict_ori = scaler.inverse_transform(y_predict)

y_test_ori = scaler.inverse_transform(y_test_scaled)

plt.figure()
plt.plot(y_test_ori, y_predict_ori, "^", color = "r")
plt.ylabel("Model Predictions")
plt.xlabel("True Values")
plt.xlim(0,5e6)
plt.ylim(0,3e6)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

k = x_test.shape[1]
n = len(x_test_scaled)

RMSE = float(format(np.sqrt(mean_squared_error(y_test_ori, y_predict_ori)) , '.3f'))
MSE = mean_squared_error(y_test_ori, y_predict_ori)
MAE = mean_absolute_error(y_test_ori, y_predict_ori)
r2 = r2_score(y_test_ori, y_predict_ori)
adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)
MAPE = np.mean( np.abs((y_test_ori - y_predict_ori) /y_test_ori ) ) * 100
print("\n7 Variables:")
print("RMSE = ", RMSE, "\nMSE =", MSE, "\nMAE =", MAE, "\nR2 =", r2, "\nAdjusted R2 =", adj_r2, "\nMAPE =", MAPE, "%")

plt.figure(figsize = (20,15))
plt.subplot(331)
plt.plot(x_test["bedrooms"], y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["bedrooms"], y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Price VS Bedrooms", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(332)
plt.plot(x_test["bathrooms"], y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["bathrooms"], y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Bathrooms")
plt.ylabel("Price")
plt.title("Price VS Bathrooms", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(333)
plt.plot(x_test["sqft_living"], y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_living"], y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Living")
plt.ylabel("Price")
plt.title("Price VS Sqft Living", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(334)
plt.plot(x_test["sqft_lot"], y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_lot"], y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Lot")
plt.ylabel("Price")
plt.title("Price VS Sqft Lot", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(335)
plt.plot(x_test["floors"], y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["floors"], y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Floors")
plt.ylabel("Price")
plt.title("Price VS Floors", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(336)
plt.plot(x_test["sqft_above"], y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_above"], y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Above")
plt.ylabel("Price")
plt.title("Price VS Sqft Above", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(337)
plt.plot(x_test["sqft_basement"], y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_basement"], y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Basement")
plt.ylabel("Price")
plt.title("Price VS Sqft Basement", weight = "bold", size = 15)
plt.legend()
plt.grid()

selected_features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", 
"floors", "waterfront","view", "condition", "grade", "sqft_above", "sqft_basement",
"yr_built", "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]

x = HouseData[selected_features]

scaler2 = MinMaxScaler()

x_scaled = scaler2.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)

x_train_scaled = scaler2.fit_transform(x_train)
x_test_scaled = scaler2.fit_transform(x_test)
y_train_scaled = scaler2.fit_transform(y_train)
y_test_scaled = scaler2.fit_transform(y_test)

model = Sequential()
model.add(Dense(100, input_dim = 18, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(1, activation = "linear"))

model.compile(optimizer = "Adam", loss = "mean_squared_error")

epochs_hist = model.fit(x_train_scaled, y_train_scaled, epochs = 100, batch_size = 50, validation_split = 0.2)

epochs_hist.history.keys()

plt.figure()
plt.plot(epochs_hist.history["loss"])
plt.plot(epochs_hist.history["val_loss"])
plt.title("Model Loss Profess During Training")
plt.xlabel("Epoch")
plt.ylabel("Training and Validation Loss")
plt.legend(["Training Loss", "Validation Loss"])

y_predict = model.predict(x_test_scaled)

plt.figure()
plt.plot(y_test, y_predict, "^", color = "r")
plt.ylabel("Model Predictions")
plt.xlabel("True Values")

y_predict_ori = scaler.inverse_transform(y_predict)

y_test_ori = scaler2.inverse_transform(y_test_scaled)

plt.figure()
plt.plot(y_test_ori, y_predict_ori, "^", color = "r")
plt.ylabel("Model Predictions")
plt.xlabel("True Values")

x_test_ori = scaler2.inverse_transform(x_test)

plt.figure(figsize = (25,25))
plt.subplot(541)
plt.plot(x_test["bedrooms"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["bedrooms"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.title("Price VS Bedrooms", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(542)
plt.plot(x_test["bathrooms"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["bathrooms"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("bathrooms")
plt.ylabel("Price")
plt.title("Price VS Bathrooms", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(543)
plt.plot(x_test["sqft_living"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_living"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Living")
plt.ylabel("Price")
plt.title("Price VS Sqft Living", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(544)
plt.plot(x_test["sqft_lot"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_lot"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Lot")
plt.ylabel("Price")
plt.title("Price VS Sqft Lot", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(545)
plt.plot(x_test["floors"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["floors"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Floors")
plt.ylabel("Price")
plt.title("Price VS Floors", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(546)
plt.plot(x_test["waterfront"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["waterfront"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Waterfront")
plt.ylabel("Price")
plt.title("Price VS Waterfront", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(547)
plt.plot(x_test["view"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["view"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("view")
plt.ylabel("Price")
plt.title("Price VS View", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(548)
plt.plot(x_test["condition"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["condition"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("condition")
plt.ylabel("Price")
plt.title("Price VS Condition", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(549)
plt.plot(x_test["grade"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["grade"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("grade")
plt.ylabel("Price")
plt.title("Price VS Grade", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,10)
plt.plot(x_test["sqft_above"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_above"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Above")
plt.ylabel("Price")
plt.title("Price VS Sqft Above", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,11)
plt.plot(x_test["sqft_basement"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_basement"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Basement")
plt.ylabel("Price")
plt.title("Price VS Sqft Basement", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,12)
plt.plot(x_test["yr_built"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["yr_built"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Yr Built")
plt.ylabel("Price")
plt.title("Price VS Yr Built", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,13)
plt.plot(x_test["yr_renovated"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["yr_renovated"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Yr Renovated")
plt.ylabel("Price")
plt.title("Price VS Yr Renovated", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,14)
plt.plot(x_test["zipcode"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["zipcode"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Zipcode")
plt.ylabel("Price")
plt.title("Price VS Zipcode", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,15)
plt.plot(x_test["lat"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["lat"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Lat")
plt.ylabel("Price")
plt.title("Price VS Lat", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,16)
plt.plot(x_test["long"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["long"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Long")
plt.ylabel("Price")
plt.title("Price VS Long", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,17)
plt.plot(x_test["sqft_living15"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_living15"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Living15")
plt.ylabel("Price")
plt.title("Price VS Sqft Living15", weight = "bold", size = 15)
plt.legend()
plt.grid()

plt.subplot(5,4,18)
plt.plot(x_test["sqft_lot15"],y_test, "o", color = "gray", label = "Data")
plt.plot(x_test["sqft_lot15"],y_predict_ori, "o", color = "red", label = "Prediction")
plt.xlabel("Sqft Lot15")
plt.ylabel("Price")
plt.title("Price VS Sqft Lot15", weight = "bold", size = 15)
plt.legend()
plt.grid()

k = x_test.shape[1]
n = len(x_test)

RMSE = float(format(np.sqrt(mean_squared_error(y_test_ori, y_predict_ori)) , '.3f'))
MSE = mean_squared_error(y_test_ori, y_predict_ori)
MAE = mean_absolute_error(y_test_ori, y_predict_ori)
r2 = r2_score(y_test_ori, y_predict_ori)
adj_r2 = 1 - (1-r2)*(n-1)/(n-k-1)
MAPE = np.mean( np.abs((y_test_ori - y_predict_ori) /y_test_ori ) ) * 100
print("\n18 Variables:")
print("RMSE = ", RMSE, "\nMSE =", MSE, "\nMAE =", MAE, "\nR2 =", r2, "\nAdjusted R2 =", adj_r2, "\nMAPE =", MAPE, "%")