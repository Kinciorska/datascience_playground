import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def visualize_iris_data_points():
    df = pd.read_csv('/Users/kinga/Documents/DataScience/svm/Iris.csv')
    df = df.drop(['Id'], axis=1)
    rows = list(range(100, 150))
    df = df.drop(df.index[rows])

    x = df['SepalLengthCm']
    y = df['PetalLengthCm']

    setosa_x = x[:50]
    setosa_y = y[:50]

    versicolor_x = x[50:]
    versicolor_y = y[50:]

    plt.figure(figsize=(8, 6))
    plt.scatter(setosa_x, setosa_y, marker='+', color='green')
    plt.scatter(versicolor_x, versicolor_y, marker='_', color='red')
    plt.show()

    plt.savefig(sys.stdout.buffer)
    sys.stdout.flush()

    # from the visualization we can deduce that a linear line can be used to separate the data points


def calculate_svm_iris():
    df = pd.read_csv('/Users/kinga/Documents/DataScience/svm/Iris.csv')
    df = df.drop(['SepalWidthCm', 'PetalWidthCm', 'Id'], axis=1)

    Y = []
    target = df['Species']
    for value in target:
        if value == 'Iris-setosa':
            Y.append(-1)
        else:
            Y.append(1)
    df = df.drop(['Species'], axis=1)
    X = list(df.values)

    X, Y = shuffle(X, Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_train = y_train[:90]
    x_train = x_train[:90]
    y_test = y_test[:10]
    x_test = x_test[:10]

    Y = y_train[:90]
    X = x_train[:90]

    svm = SVC(kernel='linear')
    svm.fit(x_train, y_train)

    y_predicted = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print(f"Accuracy score: {accuracy}")

    DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        cmap=plt.cm.Spectral,
        alpha=0.8,
        xlabel='x',
        ylabel='y',
    )

    # Scatter plot
    plt.scatter(X[:, 0], X[:, 1],
                c=Y,
                s=20, edgecolors="k")
    plt.show()

    plt.savefig(sys.stdout.buffer)
    sys.stdout.flush()


def calculate_svm_cancer():
    cancer = load_breast_cancer()
    X = cancer.data[:, :2]
    Y = cancer.target

    # Build the model
    svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
    # Trained the model
    svm.fit(X, Y)

    # Plot Decision Boundary
    DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        cmap=plt.cm.Spectral,
        alpha=0.8,
        xlabel=cancer.feature_names[0],
        ylabel=cancer.feature_names[1],
    )

    # Scatter plot
    plt.scatter(X[:, 0], X[:, 1],
                c=Y,
                s=20, edgecolors="k")
    plt.show()

    plt.savefig(sys.stdout.buffer)
    sys.stdout.flush()


if __name__ == "__main__":
    visualize_iris_data_points()
    calculate_svm_iris()
    calculate_svm_cancer()


