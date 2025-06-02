import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def handle_missing_values(df):
    return df.fillna(df.mean(numeric_only=True))  # For numeric data, fill missing values with the mean

def remove_outliers(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    return df[(z_scores < 3).all(axis=1)]  # Remove rows with any outliers

def scale_data(df):
    scaler = StandardScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df

def encode_categorical(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns)

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":

    np.random.seed(0)

    # creating a dummy dataset
    dummy_data = {
    'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],  # Normally distributed with an outlier
    'Feature2': np.random.randint(0, 100, 102).tolist(),  # Random integers
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],  # Categorical with some missing values
    'Target': np.random.choice([0, 1], 102).tolist()  # Binary target variable
    }

    # converting the dictionary to a dataframe
    df_dummy = pd.DataFrame(dummy_data)

    print(df_dummy.head())

    df_removed_missing_vals = handle_missing_values(df_dummy)

    df_removed_outliers = remove_outliers(df_removed_missing_vals)

    df_scaled_data = scale_data(df_removed_outliers)

    df_encoded_data = encode_categorical(df_scaled_data, ['Category'])

    print(df_encoded_data.head())

    # save the cleaned and preprocessed DataFrame to a CSV file for training in ml models
    save_data(df_encoded_data, 'preprocessed_dummy_data.csv')
