import pandas as pd

def load_data(path):
    """Load dataset"""
    return pd.read_csv(path)

def clean_data(df):
    """Handle missing values"""
    df = df.dropna()
    return df

def split_features_target(df):
    """Split features and target"""
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y