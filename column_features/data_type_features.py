import pandas as pd
from dateutil.parser import parse
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter, defaultdict

class DataTypeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def infer_type(self, value):
        try:
            # Try converting to float
            num = float(value)
            # If it's a whole number (e.g., 3.0), treat it as int
            if num.is_integer():
                return 'int'
            else:
                return 'float'
        except:
            pass

        try:
            # Detect booleans
            if str(value).lower() in ['true', 'false']:
                return 'bool'
        except:
            pass

        try:
            # Detect datetime
            parse(str(value))
            return 'datetime'
        except:
            pass

        return 'str'

    def majority_type(self, series):
        inferred_types = series.map(self.infer_type)
        type_counts = Counter(inferred_types)
        majority_type, _ = type_counts.most_common(1)[0]
        return majority_type

    def transform(self, X):
        dominant_types = X.apply(self.majority_type)
        #print(X.columns +" "+dominant_types)
        return dominant_types.tolist()
'''
    def transform(self, X):
        all_types = ["int", "float", "complex", "bool", "str", "datetime"]
        all_cols_types = []
        for col in X:
            type_counts = {data_type: 0 for data_type in all_types}
            type_ratios = {data_type: 0 for data_type in all_types}
            for value in X[col]:
                value_type = type(value).__name__
                if value_type not in all_types:
                    value_type = "str"
                if value_type == "str":
                    try:
                        dt = parse(value)
                        if dt is not None:
                            value_type = "datetime"
                    except Exception:
                        pass
                type_counts[value_type] += 1
            for key in type_counts:
                if type_counts[key] != 0:
                    type_ratios[key] = type_counts[key] / len(X[col])
            all_cols_types.append(type_ratios)

        type_ratios_df = pd.DataFrame(all_cols_types, index=X.columns)
        dominant_types = type_ratios_df.idxmax(axis=1)
        return dominant_types.values.tolist()
        #return pd.DataFrame(all_cols_types, index=X.columns)
'''