import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler


df = pd.read_csv('/Users/kinga/Documents/DataScience/feature_scaling/SampleFile.csv')


# absolute maximum scaling

max_vals = np.max(np.abs(df))

scaled_df = ((df - max_vals) / max_vals)


# min-max scaling

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)


# normalization

scaler = Normalizer()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)


# standarization

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)


# robust scaling

scaler = RobustScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data,
                         columns=df.columns)
