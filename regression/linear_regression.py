import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


trainig_data = pd.read_csv('/Users/kinga/Documents/DataScience/linear_regression/train.csv')
testing_data = pd.read_csv('/Users/kinga/Documents/DataScience/linear_regression/test.csv')
x_train = trainig_data['x']
y_train = trainig_data['y']
x_test = testing_data['x']
y_test = testing_data['y']

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# Calculate linear regression manually

n = 700
alpha = 0.0001

alpha_0 = np.zeros((n, 1))
alpha_1 = np.zeros((n, 1))

epochs = 0
while epochs < 1000:
    y = alpha_0 + alpha_1 * x_train
    error = y - y_train
    alpha_0 = alpha_0 - (alpha * 2 / n * np.sum(error))
    alpha_1 = alpha_1 - (alpha * 2 / n * np.sum(error * x_train))

    epochs += 1

alpha_0 = alpha_0[:300]
alpha_1 = alpha_1[:300]

y_predicted = alpha_0 + alpha_1 * x_test

model_accuracy = r2_score(y_test, y_predicted)

print(f"R2 score: {model_accuracy}")

y_plot = []
for i in range(100):
    y_plot.append(alpha_0 + alpha_1 * i)
x_plot = range(len(y_plot))
y_plot = np.array(y_plot)
y_plot = y_plot.reshape(100, 300)
plt.figure(figsize=(10, 10))
plt.scatter(x_test, y_test, color='red')
plt.plot(x_plot, y_plot, color='black', label='predicted')
plt.show()

plt.savefig(sys.stdout.buffer)
sys.stdout.flush()


# Calculate linear regression using sklearn

clf = LinearRegression().fit(x_train, y_train)
y_predicted = clf.predict(x_test)

model_accuracy = r2_score(y_test, y_predicted)

print(f"R2 score: {model_accuracy}")

# there is a possibility to use the linear regression model after using non-linear models, i.e. Polynomial regression
