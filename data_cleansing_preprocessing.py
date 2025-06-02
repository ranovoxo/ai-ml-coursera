import pandas as pd
import numpy as np
import missingno as msno  # Optional: for visualizing missing data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

# load dataset into a pandas DataFrame
df = pd.read_csv('customer_data.csv')  

# display the first few rows of the dataset
print(f"{df.head()}\n\n")


df['Age'].replace(r'^\s*$', pd.NA, regex=True, inplace=True)  # replace spaces with NaN
df['Age'].fillna("None", inplace=True)  # fill NaNs with 'None'


df['Name'].fillna("Unknown", inplace=True) # fill missing Name with "Unknown"

df['Income'].fillna(df['Income'].median(), inplace=True) # fill missing Income with median

df.reset_index(drop=True, inplace=True) # reset index after cleanup

print(df)

# Identify outliers using Z-score

# keep only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# compute Z-scores on numeric columns only because the function can only handle numerica data
z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))

df_no_outliers = numeric_df[(z_scores < 3).all(axis=1)]

print("Z-Scores: ", z_scores)
print("DF-Num_Outliers: ", df_no_outliers)

# Or cap outliers at a threshold
upper_limit = df['Income'].quantile(0.95)
df['Income'] = np.where(df['Income'] > upper_limit, upper_limit, df['Income'])

print(df)

# min-max scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

# z-score Standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

# one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_scaled, columns=['Income'])

print(df_encoded)

df_encoded.to_csv('cleaned_preprocessed_data.csv', index=False)


