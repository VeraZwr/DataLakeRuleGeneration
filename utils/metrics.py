from sklearn.metrics import precision_score, recall_score, f1_score

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

from collections import defaultdict

def compute_actual_errors(clean_dataset_dict, dirty_dataset_dict):
    from collections import defaultdict
    actual_errors_by_column = defaultdict(list)

    for table_name in clean_dataset_dict:
        clean_df = clean_dataset_dict[table_name]
        dirty_df = dirty_dataset_dict.get(table_name)

        if dirty_df is None:
            continue

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

