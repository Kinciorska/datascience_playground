import numpy as np
import pandas as pd

df = pd.read_csv('/Users/kinga/Documents/DataScience/feature_scaling/SampleFile.csv')


# absolute maximum scaling

max_vals = np.max(np.abs(df))

scaled_df = ((df - max_vals) / max_vals)
