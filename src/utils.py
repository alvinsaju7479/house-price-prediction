
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_features_target(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

