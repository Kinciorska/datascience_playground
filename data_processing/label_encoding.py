import pandas as pd

from sklearn import preprocessing

df = pd.read_csv('/Users/kinga/Documents/DataScience/Iris.csv')

label_encoder = preprocessing.LabelEncoder()


# encode labels in column 'Species'
df['Species'] = label_encoder.fit_transform(df['Species'])

# the issue with label encoding is that it may lead to priority issues
# because the model may consider higher values as higher priority
