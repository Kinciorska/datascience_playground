import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.datasets import make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,  precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


df = pd.read_csv('/Users/kinga/Documents/DataScience/inbalanced_data_handling/creditcard.csv')

# normalise the amount column
df['normAmount'] = StandardScaler().fit_transform(np.array(df['Amount']).reshape(-1, 1))

# drop Time and Amount columns as they are not relevant for prediction purposes
df = df.drop(['Time', 'Amount'], axis=1)

# check the number of fraud transactions
n_o_fraud_transactions = df['Class'].value_counts()


X = df.drop('Class', axis=1).values
y = df['Class'].values

# split the data into train and test sets in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# logistic regression object
lr = LogisticRegression()

# train the model on train set
lr.fit(X_train, y_train.ravel())

predictions = lr.predict(X_test)

# check classification report
print(classification_report(y_test, predictions))

# the recall of the minority class is noticeably lower
# we proved that the model is more biased towards the majority class
# we need to apply imbalance handling techniques

# SMOTE balances class distribution by randomly increasing minority class examples by replicating them

sm = SMOTE(random_state=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train.ravel())

lr1 = LogisticRegression()
lr1.fit(X_train_sm, y_train_sm.ravel())
predictions_after_smote = lr1.predict(X_test)

# print classification report
print(classification_report(y_test, predictions_after_smote))

# Near miss balances class distribution by randomly eliminating majority class examples

nm = NearMiss()
X_train_nm, y_train_nm = nm.fit_resample(X_train, y_train.ravel())

lr2 = LogisticRegression()
lr2.fit(X_train_nm, y_train_nm.ravel())
predictions_after_near_miss = lr2.predict(X_test)

print(classification_report(y_test, predictions_after_near_miss))


# the BalancedBaggingClassifier balances training by incorporating a parameter "sampling strategy"
# which determines the type of resampling

# create an imbalanced dataset

X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

base_classifier = RandomForestClassifier(random_state=42)

balanced_bagging_classifier = BalancedBaggingClassifier(base_classifier,
                                                        sampling_strategy='auto',
                                                        replacement=False,
                                                        random_state=42)

balanced_bagging_classifier.fit(X_train, y_train)


y_predicted = balanced_bagging_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_predicted)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_predicted))


# Changing the threshold to a calcuated value (default is 0.5) allows to better predict the correct class membership

# create an imbalanced dataset

X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# predict probabilities on the test set
y_proba = model.predict_proba(X_test)[:, 1]

threshold = 0.5

# adjust threshold based on your criteria (e.g., maximizing F1-score)
while threshold >= 0:
    y_predicted = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_predicted)

    print(f"Threshold: {threshold:.2f}, F1 Score: {f1:.4f}")

    # Move the threshold (you can customize the step size)
    threshold -= 0.02

# Outlier Detection algorithm can be used to identify the rare datapoints in the dataset,
# assume majority class data as normal and minority class data as outlier
# train the model only on the majority class data allows the model to predict if the record is normal or outlier

# Local Outlier Factor (LOF)


df = load_iris(as_frame=True).frame

X = df[['sepal length (cm)', 'sepal width (cm)']]

# define the model and set the number of neighbors
model = LocalOutlierFactor(n_neighbors=5)

model.fit(X)

# calculate the outlier scores for each point
scores = model.negative_outlier_factor_

# identify the points with the highest outlier scores
outliers = np.argwhere(scores > np.percentile(scores, 95))

colors = ['green', 'red']


for i in range(len(X)):
    if i not in outliers:
        plt.scatter(X.iloc[i,0], X.iloc[i,1], color=colors[0])  # Not anomly
    else:
        plt.scatter(X.iloc[i,0], X.iloc[i,1], color=colors[1])  # anomly
plt.xlabel('sepal length (cm)',fontsize=13)
plt.ylabel('sepal width (cm)',fontsize=13)
plt.title('Anomly by Local Outlier Factor', fontsize=16)
plt.show()

plt.savefig(sys.stdout.buffer)
sys.stdout.flush()


# Isolation Forest

df = load_iris(as_frame=True).frame
X = df[['sepal length (cm)', 'sepal width (cm)']]

# define the model and set the contamination level
model = IsolationForest(contamination=0.05)

model.fit(X)

# calculate the outlier scores for each point
scores = model.decision_function(X)

# identify the points with the highest outlier scores
outliers = np.argwhere(scores < np.percentile(scores, 5))


colors = ['green', 'red']

for i in range(len(X)):
    if i not in outliers:
        plt.scatter(X.iloc[i, 0], X.iloc[i, 1], color=colors[0])  # Not anomly
    else:
        plt.scatter(X.iloc[i, 0], X.iloc[i, 1], color=colors[1])  # anomly
plt.xlabel('sepal length (cm)', fontsize=13)
plt.ylabel('sepal width (cm)', fontsize=13)
plt.title('Anomly by Isolation Forest', fontsize=16)
plt.show()


# using Tree base models gives better results in case of inbalanced datasets than non tree-based models
#     -Decision Trees
#     -Random Forests
#     -Gradien Boosted Trees

