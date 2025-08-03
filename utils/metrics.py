from sklearn.metrics import precision_score, recall_score, f1_score
from rules.evaluation import detect_combined_errors, detect_dynamic_errors
from utils.file_io import csv_to_column_dict
from pathlib import Path
import pandas as pd
from collections import defaultdict
from utils.read_data import read_csv

def merge_errors(errors):
    merged = defaultdict(set)
    for err in errors:
        merged[(err["table"], err["column"])].update(err["error_indices"])

    return [
        {"table": tbl, "column": col, "error_indices": sorted(list(indices))}
        for (tbl, col), indices in merged.items()
    ]

def compute_cell_level_scores(errors, raw_dataset, clean_dataset_dict):
    predicted_errors = set()
    actual_errors = set()

    for table, dirty_df in raw_dataset.items():
        clean_df = clean_dataset_dict[table]

        # Ensure same row count
        dirty_df = dirty_df.reset_index(drop=True)
        clean_df = clean_df.reset_index(drop=True)

        # Map column names in dirty_df to their index
        col_index_map = {name: idx for idx, name in enumerate(dirty_df.columns)}

        # --- Build predicted cells ---
        for err in errors:
            if err["table"] != table:
                continue
            for idx in err["error_indices"]:
                col_idx = col_index_map.get(err["column"])
                if col_idx is not None:
                    predicted_errors.add((table, col_idx, idx))

        # --- Build actual cells (dirty vs clean) ---
        for col_idx in range(len(dirty_df.columns)):
            dirty_col = dirty_df.iloc[:, col_idx]
            clean_col = clean_df.iloc[:, col_idx]

            diffs = dirty_col != clean_col
            for idx in dirty_df[diffs].index:
                actual_errors.add((table, col_idx, idx))

    # --- Calculate metrics ---
    TP = len(predicted_errors & actual_errors)
    FP = len(predicted_errors - actual_errors)
    FN = len(actual_errors - predicted_errors)

    precision = TP / (TP + FP) if predicted_errors else 0
    recall = TP / (TP + FN) if actual_errors else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f"[DEBUG] Predicted errors: {len(predicted_errors)}, Actual errors: {len(actual_errors)}")
    return precision, recall, f1


def compute_actual_errors(clean_dataset_dict, dirty_dataset_dict):
    actual_errors_by_column = defaultdict(list)

    for table_name in clean_dataset_dict:
        clean_df = clean_dataset_dict[table_name]
        if clean_df is None:
            continue
        dirty_df = dirty_dataset_dict.get(table_name)
        if dirty_df is None:
            continue
        clean_df = clean_df.reset_index(drop=True)
        dirty_df = dirty_df.reset_index(drop=True)

        # Find common columns
        min_cols = min(clean_df.shape[1], dirty_df.shape[1])
        common_columns = set(clean_df.columns).intersection(dirty_df.columns)

        if clean_df.shape[0] != dirty_df.shape[0]:
            print(f"Row count mismatch in table '{table_name}', skipping")
            continue

        for row_idx in range(len(clean_df)):
            for col_idx in range(min_cols):
                try:
                    #print(f"COlUMN_test {dirty_df.columns[col_idx]}{row_idx}")
                    clean_val = str(clean_df.iat[row_idx, col_idx])
                    #print(clean_val)
                    dirty_val = str(dirty_df.iat[row_idx, col_idx])
                    #print(dirty_val)
                    if clean_val != dirty_val:
                        col_name_dirty = dirty_df.columns[col_idx]
                        #print(f"Debug:  ")
                        actual_errors_by_column[(table_name, col_name_dirty)].append(row_idx)
                except Exception as e:
                    print(f"Error comparing cell [{row_idx}, {col_name_dirty}] in table '{table_name}': {e}")

    return actual_errors_by_column


def evaluate_one_dataset_only(rules, shared_rules, clusters, column_profiles, dataset_name):

    dataset_path = Path("datasets/Quintet") / dataset_name
    # Load dirty and clean data
    dirty_df = read_csv(dataset_path / "dirty.csv")
    clean_df = read_csv(dataset_path / "clean.csv")

    raw_dataset = {dataset_name: dirty_df}
    clean_dataset_dict = {dataset_name: clean_df}

    # Detect errors using the combined approach
    print(f"\n Detecting errors for {dataset_name}...")
    errors = detect_combined_errors(clusters, shared_rules, rules, raw_dataset, column_profiles)
    errors = merge_errors(errors)
    # Print out error counts and values
    for err in errors:
        if err['table'] == dataset_name:
            err_count = len(err['error_indices'])
            err_values = [dirty_df.at[idx, err['column']] for idx in err['error_indices']]
            print(f"Table: {err['table']} | Column: {err['column']} | Error count: {err_count}")
            print(f"Error rows: {err['error_indices']}")
            print(f"Error values: {err_values}\n")

    # Compute evaluation metrics
    precision, recall, f1 = compute_cell_level_scores(errors, raw_dataset, clean_dataset_dict)
    print(f"📊 Evaluation Metrics for {dataset_name}:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    # Optionally: Print actual error count per column
    actual_errors_by_column = compute_actual_errors(clean_dataset_dict, raw_dataset)
    print("\n Actual Error Counts (by column):")
    for (table, col), indices in actual_errors_by_column.items():
        if table == dataset_name:
            print(f"Table: {table} | Column: {col} | Actual Error Count: {len(indices)}")


