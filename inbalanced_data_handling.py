import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


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
# we need to apply imbalance handling techniques like SMOTE or Near miss

# SMOTE balances class distribution by randomly increasing minority class examples by replicating them

sm = SMOTE(random_state=2)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train.ravel())

lr1 = LogisticRegression()
lr1.fit(X_train_sm, y_train_sm.ravel())
predictions_after_smote = lr1.predict(X_test)

# print classification report
print(classification_report(y_test, predictions_after_smote))
