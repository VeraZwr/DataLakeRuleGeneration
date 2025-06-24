import pickle
import json
import pandas as pd
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def csv_to_column_dict(csv_path):
    df = pd.read_csv(csv_path, dtype=str).fillna("empty")  # force string, handle nulls
    col_dict = {col: df[col].tolist() for col in df.columns}
    return col_dict
"""
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path, indent=2):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
"""