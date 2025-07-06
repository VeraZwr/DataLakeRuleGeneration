from doduo.doduo.doduo import Doduo, DFColTypeColwiseDataset
import argparse
import pandas as pd

# Load Doduo model
args = argparse.Namespace()
args.model = "viznet"
doduo = Doduo(args)

# Load your CSV
df1 = pd.read_csv("/Users/veraz/PycharmProjects/DataLakeRuleGeneration/datasets/Quintet/hospital/dirty.csv", index_col=0)

# Run annotate_columns
annot_df1 = doduo.annotate_columns(df1)
print("Predicted coltypes:", annot_df1.coltypes)
print("Valid col indices:", annot_df1.valid_col_indices)
print("Total input columns:", df1.columns.tolist())

# Re-check the dataset’s filter logic
input_dataset = DFColTypeColwiseDataset(df1, doduo.tokenizer)
print("Manually created valid_col_indices:", input_dataset.valid_col_indices)

# Show pairs, if any
for col_index, coltype in zip(annot_df1.valid_col_indices, annot_df1.coltypes):
    print(f"Column: {df1.columns[col_index]}, Predicted Type: {coltype}")
else:
    print("⚠️  No valid columns were detected by DFColTypeColwiseDataset.")
