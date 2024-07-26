import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Building a dummy employee dataset for example

data = {'Employee id': [10, 20, 15, 25, 30],
        'Gender': ['M', 'F', 'F', 'M', 'F'],
        'Remarks': ['Good', 'Nice', 'Good', 'Great', 'Nice'],
        }

df = pd.DataFrame(data)

print(f"Employee data : \n{df}")

# Extract categorical (object type) columns from the dataframe
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(df[categorical_columns])

# Create a DataFrame with the one-hot encoded columns
# Use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
df_encoded = pd.concat([df, one_hot_df], axis=1)

# Drop the original categorical columns
df_encoded = df_encoded.drop(categorical_columns, axis=1)

# In case of binary categories - i.e. male, female, only one column is enough, the data can be collected
# while there will be no redundancy and dummy variable trap

# Display the resulting dataframe
print(f"Encoded Employee data : \n{df_encoded}")
