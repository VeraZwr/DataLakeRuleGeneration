import argparse
import pandas as pd
from doduo.doduo.doduo import Doduo

# Load Doduo model
args = argparse.Namespace()
args.model = "viznet"
doduo = Doduo(args)

# Load sample tables
df1 = pd.read_csv("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/datasets/Quintet/hospital/dirty.csv", index_col=0)

# Sample 1: Column annotation
annot_df1 = doduo.annotate_columns(df1)
print(annot_df1.coltypes)

# Print column names with corresponding predicted column types (coltypes)
for column_name, coltype in zip(df1.columns, annot_df1.coltypes):
    print(f"Column: {column_name}, Predicted Type: {coltype}")