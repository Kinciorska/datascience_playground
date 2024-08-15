import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from util import sigmoid

df = pd.read_csv('/Users/kinga/Documents/DataScience/non-linear_regression/gdp.csv')

# check the data - year (independent variable) vs value (dependent variable)
plt.figure(figsize=(8, 5))
x_original, y_original = df["Year"].values, df["Value"].values
plt.plot(x_original, y_original, 'bo')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title('Original GDP Data')
plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()

# it is similar to a simple logistic model curve (it will be even more after fitting it):
X_logistic = np.arange(-5.0, 5.0, 0.1)
Y_logistic = 1.0 / (1.0 + np.exp(-X_logistic))

plt.plot(X_logistic, Y_logistic, color='green')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.title('Simple Logistic Model Curve')
plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()

# set initial values for logistic function parameters
beta_1_initial = 0.10
beta_2_initial = 1990.0

# compare logistic function to the data
Y_pred_logistic = sigmoid(x_original, beta_1_initial, beta_2_initial)

plt.plot(x_original, Y_pred_logistic * 15000000000000., color='purple', label='Initial Prediction')
plt.plot(x_original, y_original, 'go', label='Data')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.legend()
plt.title('Initial Logistic Regression Fit')
plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()

# normalize the data to scale between 0 and 1
x_normalized = x_original / max(x_original)
y_normalized = y_original / max(y_original)

# we can use curve_fit function from scipy.optimize
# to find the best-fitting parameters for the sigmoid function based on the normalized data

popt, pcov = curve_fit(sigmoid, x_normalized, y_normalized)

# print the final parameters
print("Beta_1 = %f, Beta_2 = %f" % (popt[0], popt[1]))

# Beta_1 = 690.451712, Beta_2 = 0.997207

# create a new x range for the fitted sigmoid curve
x_fit = np.linspace(1960, 2015, 55) / max(x_original)

# apply the sigmoid function with the fitted parameters
y_fit = sigmoid(x_fit, *popt)

# plot the normalized data and the sigmoid fit
plt.figure(figsize=(8, 5))
plt.plot(x_normalized, y_normalized, 'go', label='Normalized Data')
plt.plot(x_fit, y_fit, linewidth=3.0, color='purple',
         label='Sigmoid Fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.title('Normalized Sigmoid Regression Fit')
plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()

# now we have it fitted

# split data into train/test sets
random_mask = np.random.rand(len(df)) < 0.8
train_x = x_normalized[random_mask]
test_x = x_normalized[~random_mask]
train_y = y_normalized[random_mask]
test_y = y_normalized[~random_mask]

# build the model using the train set
popt_train, pcov_train = curve_fit(sigmoid, train_x, train_y)

# predict using the test set
y_hat_test = sigmoid(test_x, *popt_train)

# evaluate the model
mean_absolute_err = mean_absolute_error(test_y, y_hat_test)
mean_square_err = np.mean((y_hat_test - test_y) ** 2)
r2 = r2_score(y_hat_test, test_y)

# print the evaluation metrics
print("Mean Absolute Error: %.2f" % mean_absolute_err)
print("Mean Squared Error: %.2f" % mean_square_err)
print("R2-score: %.2f" % r2)
