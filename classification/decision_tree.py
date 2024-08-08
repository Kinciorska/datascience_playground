import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
df = pd.read_csv(url, sep=',', header=None)

# display data info
print("Dataset Length: ", len(df))
print("Dataset Shape: ", df.shape)
print("Dataset: ", df.head())

# separate the target variable
X = df.values[:, 1:5]
y = df.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=100)


# train using Gini Index

# define the model
model = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

confusion = confusion_matrix(y_test, y_predicted)

accuracy = accuracy_score(y_test, y_predicted)

report = classification_report(y_test, y_predicted)

print("Gini Index")
print(f"Predicted values: {y_predicted}")
print(f"Confusion Matrix: {confusion}")
print(f"Accuracy: {accuracy}")
print(f"Report: {report}")


# train using Entropy

# define the model
model = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

confusion = confusion_matrix(y_test, y_predicted)

accuracy = accuracy_score(y_test, y_predicted)

report = classification_report(y_test, y_predicted)

print("Entropy")
print(f"Predicted values: {y_predicted}")
print(f"Confusion Matrix: {confusion}")
print(f"Accuracy: {accuracy}")
print(f"Report: {report}")
