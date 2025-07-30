from sklearn.metrics import precision_score, recall_score, f1_score
from rules.evaluation import detect_combined_errors, detect_dynamic_errors
from utils.file_io import csv_to_column_dict
from pathlib import Path
import pandas as pd
from collections import defaultdict


def compute_cell_level_scores(errors, raw_dataset, clean_dataset_dict):
    predicted_errors = set()
    for err in errors:
        table = err['table']
        column = err['column']
        for row_idx in err['error_indices']:
            predicted_errors.add((table, row_idx, column))

    # Build ground truth
    ground_truth_errors = set()
    for table, clean_df in clean_dataset_dict.items():
        dirty_df = raw_dataset[table]
        for col in clean_df.columns:
            if col not in dirty_df.columns:
                continue  # Skip if dirty data is missing this column
            for idx in range(min(len(clean_df), len(dirty_df))):
                clean_val = str(clean_df.at[idx, col])
                dirty_val = str(dirty_df.at[idx, col])
                if clean_val != dirty_val:
                    ground_truth_errors.add((table, idx, col))

    # Construct binary vectors
    all_cells = predicted_errors.union(ground_truth_errors)
    y_true = [1 if cell in ground_truth_errors else 0 for cell in all_cells]
    y_pred = [1 if cell in predicted_errors else 0 for cell in all_cells]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def compute_actual_errors(clean_dataset_dict, dirty_dataset_dict):
    actual_errors_by_column = defaultdict(list)

    for table_name in clean_dataset_dict:
        clean_df = clean_dataset_dict[table_name]
        if clean_df is None:
            continue
        dirty_df = dirty_dataset_dict.get(table_name)
        clean_df = clean_df.reset_index(drop=True)
        dirty_df = dirty_df.reset_index(drop=True)

        # Find common columns
        common_columns = set(clean_df.columns).intersection(dirty_df.columns)

        if clean_df.shape[0] != dirty_df.shape[0]:
            print(f"Row count mismatch in table '{table_name}', skipping")
            continue

        for row_idx in range(len(clean_df)):
            for col in common_columns:
                try:
                    clean_val = str(clean_df.at[row_idx, col])
                    dirty_val = str(dirty_df.at[row_idx, col])
                    if clean_val != dirty_val:
                        actual_errors_by_column[(table_name, col)].append(row_idx)
                except Exception as e:
                    print(f"Error comparing cell [{row_idx}, {col}] in table '{table_name}': {e}")

    return actual_errors_by_column


def evaluate_one_dataset_only(rules, shared_rules, clusters, column_profiles):
    dataset_name = "beers"
    dataset_path = Path("datasets/Quintet") / dataset_name

    # Load dirty and clean data
    dirty_df = pd.read_csv(dataset_path / "dirty.csv")
    clean_df = pd.read_csv(dataset_path / "clean.csv")

    raw_dataset = {dataset_name: dirty_df}
    clean_dataset_dict = {dataset_name: clean_df}

    # Detect errors using the combined approach
    print(f"\n Detecting errors for {dataset_name}...")
    errors = detect_combined_errors(clusters, shared_rules, rules, raw_dataset, column_profiles)

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


