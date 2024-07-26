import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from util import sigmoid

df = pd.read_csv('/Users/kinga/Documents/DataScience/logistic_regression/Iris.csv')
df = df.drop(['Id'], axis=1)
rows = list(range(100, 150))
df = df.drop(df.index[rows])  # Drop the rows with target values Iris-virginica
Y = []
target = df['Species']
for val in target:
    if val == 'Iris-setosa':
        Y.append(0)
    else:
        Y.append(1)
df = df.drop(['Species'], axis=1)
X = df.values.tolist()

X, Y = shuffle(X, Y)

x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_1 = x_train[:, 0]
x_2 = x_train[:, 1]
x_3 = x_train[:, 2]
x_4 = x_train[:, 3]

x_1 = np.array(x_1)
x_2 = np.array(x_2)
x_3 = np.array(x_3)
x_4 = np.array(x_4)

x_1 = x_1.reshape(90, 1)
x_2 = x_2.reshape(90, 1)
x_3 = x_3.reshape(90,  1)
x_4 = x_4.reshape(90, 1)

y_train = y_train.reshape(90, 1)

m = 90
alpha = 0.0001

theta_0 = np.zeros((m, 1))
theta_1 = np.zeros((m, 1))
theta_2 = np.zeros((m, 1))
theta_3 = np.zeros((m, 1))
theta_4 = np.zeros((m, 1))


def calculate_logistic_regression(x_train, y_train):
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_predicted = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print(accuracy)


def calculate_logistic_regression_manually(theta_0, theta_1, theta_2, theta_3, theta_4):
    epochs = 0
    cost_func = []

    while epochs < 10000:
        y = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * x_3 + theta_4 * x_4
        y = sigmoid(y)

        cost = (- np.dot(np.transpose(y_train), np.log(y)) - np.dot(np.transpose(1 - y_train), np.log(1 - y))) / m

        theta_0_grad = np.dot(np.ones((1, m)), y - y_train) / m
        theta_1_grad = np.dot(np.transpose(x_1), y - y_train) / m
        theta_2_grad = np.dot(np.transpose(x_2), y - y_train) / m
        theta_3_grad = np.dot(np.transpose(x_3), y - y_train) / m
        theta_4_grad = np.dot(np.transpose(x_4), y - y_train) / m

        theta_0 = theta_0 - alpha * theta_0_grad
        theta_1 = theta_1 - alpha * theta_1_grad
        theta_2 = theta_2 - alpha * theta_2_grad
        theta_3 = theta_3 - alpha * theta_3_grad
        theta_4 = theta_4 - alpha * theta_4_grad

        cost_func.append(cost)
        epochs += 1

    test_x_1 = np.array(x_test[:, 0])
    test_x_2 = np.array(x_test[:, 1])
    test_x_3 = np.array(x_test[:, 2])
    test_x_4 = np.array(x_test[:, 3])

    test_x_1 = test_x_1.reshape(10, 1)
    test_x_2 = test_x_2.reshape(10, 1)
    test_x_3 = test_x_3.reshape(10, 1)
    test_x_4 = test_x_4.reshape(10, 1)

    index = list(range(10, 90))

    theta_0 = np.delete(theta_0, index)
    theta_1 = np.delete(theta_1, index)
    theta_2 = np.delete(theta_2, index)
    theta_3 = np.delete(theta_3, index)
    theta_4 = np.delete(theta_4, index)

    theta_0 = theta_0.reshape(10, 1)
    theta_1 = theta_1.reshape(10, 1)
    theta_2 = theta_2.reshape(10, 1)
    theta_3 = theta_3.reshape(10, 1)
    theta_4 = theta_4.reshape(10, 1)

    y_predicted = theta_0 + theta_1 * test_x_1 + theta_2 * test_x_2 + theta_3 * test_x_3 + theta_4 * test_x_4
    y_predicted = sigmoid(y_predicted)

    new_y_predicted = []
    for value in y_predicted:
        if value >= 0.5:
            new_y_predicted.append(1)
        else:
            new_y_predicted.append(0)
    accuracy = accuracy_score(y_test, new_y_predicted)
    print(accuracy)


if __name__ == "__main__":
    calculate_logistic_regression(x_train, y_train)
    calculate_logistic_regression_manually(theta_0, theta_1, theta_2, theta_3, theta_4)
