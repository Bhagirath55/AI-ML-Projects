import numpy as np
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df, numeric_strategy='mean', drop_na=False):
    """  Handles missing values in the dataset based on user choice.
     - `numeric_strategy`: Choose "mean", "median", or "mode" for numeric columns.
      - `drop_na`: If True, drops all rows with missing values.  """
    df = df.copy()
    if drop_na:
        return df.dropna()

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Handle Numeric Columns
    if numeric_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif numeric_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif numeric_strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])

    # Handle categorical columns (always filling with 'Unknown')
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    return df


def remove_outliers(df):
    """Remove outliers using Inter quartile Range (IQR) Method"""

    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    return df


def scale_features(df):
    """Scale Numeric  Features using StandardScaler"""

    df = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
