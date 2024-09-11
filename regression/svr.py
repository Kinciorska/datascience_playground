import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('/Users/kinga/Documents/DataScience/non-linear_regression/svr.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# feature scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = sc_x.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

y_train = np.ravel(y_train)

# define the model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

y_predicted = sc_y.inverse_transform(model.predict(sc_x.transform(X_test)).reshape(-1, 1))

# make predictions
np.set_printoptions(precision=2)
print(f"Predicted values: {y_predicted}")

# check performance
performance = r2_score(y_predicted, y_test)
print(f"Performance: {performance}")

