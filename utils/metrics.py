from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score
from rules.evaluation import detect_combined_errors, detect_dynamic_errors
from utils.file_io import csv_to_column_dict
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
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
        actual_errors_by_column = compute_actual_errors(clean_dataset_dict, raw_dataset)
        for (table, col_name), row_indices in actual_errors_by_column.items():
            dirty_df = raw_dataset.get(table)
            if dirty_df is None:
                continue

            # Map column name to index
            col_index_map = {name: idx for idx, name in enumerate(dirty_df.columns)}
            col_idx = col_index_map.get(col_name)
            if col_idx is None:
                continue

            for row_idx in row_indices:
                actual_errors.add((table, col_idx, row_idx))
    # --- Calculate metrics ---
    print("predicted errors:", len(predicted_errors))
    print("actual errors:", len(actual_errors))
    TP = len(predicted_errors & actual_errors)
    FP = len(predicted_errors - actual_errors)
    FN = len(actual_errors - predicted_errors)

    precision = TP / (TP + FP) if predicted_errors else 0
    recall = TP / (TP + FN) if actual_errors else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f"[DEBUG] Predicted cells: {len(predicted_errors)}, Actual cells: {len(actual_errors)}")
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


def evaluate_one_dataset_only(rules, shared_rules, clusters, column_profiles, dataset_group, dataset_name, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = Path("datasets") / dataset_group / dataset_name
    dirty_df = read_csv(dataset_path / "dirty.csv")
    clean_df = read_csv(dataset_path / "clean.csv")

    raw_dataset = {dataset_name: dirty_df}
    clean_dataset_dict = {dataset_name: clean_df}

    print(f"\n Detecting errors for {dataset_name}...")
    errors = detect_combined_errors(clusters, shared_rules, rules, raw_dataset, column_profiles)
    errors = merge_errors(errors)

    # Print detected errors
    for err in errors:
        if err['table'] == dataset_name:
            err_count = len(err['error_indices'])
            err_values = [dirty_df.at[idx, err['column']] for idx in err['error_indices']]
            print(f"Table: {err['table']} | Column: {err['column']} | Error count: {err_count}")
            print(f"Error rows: {err['error_indices']}")
            print(f"Error values: {err_values}\n")

    # --- Compute TP, FP, FN, TN ---
    predicted = set((dataset_name, col, row)
                    for err in errors for row in err["error_indices"]
                    for col in [err["column"]])
    actual = set((dataset_name, col, row)
                 for (tbl, col), rows in compute_actual_errors(clean_dataset_dict, raw_dataset).items()
                 if tbl == dataset_name for row in rows)

    TP = len(predicted & actual)
    FP = len(predicted - actual)
    FN = len(actual - predicted)
    total_cells = dirty_df.shape[0] * dirty_df.shape[1]
    TN = total_cells - (TP + FP + FN)

    # --- Metrics ---
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f" Evaluation Metrics for {dataset_name}:")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    # --- Actual error counts ---
    actual_errors_by_column = compute_actual_errors(clean_dataset_dict, raw_dataset)
    print("\n Actual Error Counts (by column):")
    for (table, col), indices in actual_errors_by_column.items():
        if table == dataset_name:
            print(f"Table: {table} | Column: {col} | Actual Error Count: {len(indices)}")
    total_actual_errors = sum(
        len(indices)
        for (table, _), indices in actual_errors_by_column.items()
        if table == dataset_name
    )
    print("Total actual cell errors:", total_actual_errors)

    # --- Write metrics to file ---
    output_file = Path("output") /dataset_group/ f"evaluation_results_{dataset_name}_{timestamp}.txt"
    with open(output_file, "w") as f:
        f.write(f"===== Metrics for {dataset_name} =====\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"TP: {TP}\n")
        f.write(f"FP: {FP}\n")
        f.write(f"FN: {FN}\n")
        f.write(f"TN: {TN}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall: {recall:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n")
        f.write("\nActual Error Counts (by column):\n")
        for (table, col), indices in actual_errors_by_column.items():
            if table == dataset_name:
                f.write(f"Table: {table} | Column: {col} | Actual Error Count: {len(indices)}\n")
        f.write(f"\n Total actual cell errors: {total_actual_errors}\n")

    print(f"\n Results have been saved to {output_file}")


def evaluate_multiple_datasets(rules, shared_rules, clusters, column_profiles, dataset_group, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_TP = overall_FP = overall_FN = overall_TN = 0
    output_dir = Path("output") / dataset_group
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"evaluation_results_{dataset_group}_{timestamp}.txt"

    dataset_dir = Path("datasets") / dataset_group
    dataset_names = [
        subfolder.name
        for subfolder in dataset_dir.iterdir()
        if subfolder.is_dir() and not subfolder.name.startswith(".")
    ]

    with open(output_file, "w") as f:
        for dataset_name in dataset_names:
            dataset_path = dataset_dir / dataset_name
            dirty_file = dataset_path / "dirty.csv"
            clean_file = dataset_path / "clean.csv"

            if not dirty_file.exists() or not clean_file.exists():
                msg = f"Skipping {dataset_name} (missing files)\n"
                print(msg.strip())
                f.write(msg)
                continue

            dirty_df = read_csv(dirty_file)
            clean_df = read_csv(clean_file)
            raw_dataset = {dataset_name: dirty_df}
            clean_dataset_dict = {dataset_name: clean_df}

            errors = detect_combined_errors(clusters, shared_rules, rules, raw_dataset, column_profiles)
            errors = merge_errors(errors)

            predicted = set((dataset_name, col, row)
                            for err in errors for row in err["error_indices"]
                            for col in [err["column"]])
            actual = set((dataset_name, col, row)
                         for (tbl, col), rows in compute_actual_errors(clean_dataset_dict, raw_dataset).items()
                         if tbl == dataset_name for row in rows)

            TP = len(predicted & actual)
            FP = len(predicted - actual)
            FN = len(actual - predicted)
            total_cells = dirty_df.shape[0] * dirty_df.shape[1]
            TN = total_cells - (TP + FP + FN)

            precision = TP / (TP + FP) if TP + FP else 0
            recall = TP / (TP + FN) if TP + FN else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

            msg = (
                f"\nDataset Metrics for {dataset_name}:\n"
                f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}\n"
                f"Precision: {precision:.3f}\n"
                f"Recall: {recall:.3f}\n"
                f"F1 Score: {f1:.3f}\n"
                f"--- Per-Column Metrics ---\n"
            )
            print(msg)
            f.write(msg)

            # --- Per-column metrics ---
            predicted_cells = defaultdict(set)
            actual_cells = defaultdict(set)

            col_index_map = {name: idx for idx, name in enumerate(dirty_df.columns)}
            for err in errors:
                if err["table"] == dataset_name:
                    col_idx = col_index_map.get(err["column"])
                    if col_idx is not None:
                        for idx in err["error_indices"]:
                            predicted_cells[col_idx].add(idx)

            actual_errors_by_column = compute_actual_errors(clean_dataset_dict, raw_dataset)
            for (table, col_name), row_indices in actual_errors_by_column.items():
                if table == dataset_name:
                    col_idx = col_index_map.get(col_name)
                    if col_idx is not None:
                        for row_idx in row_indices:
                            actual_cells[col_idx].add(row_idx)

            for col_idx, col_name in enumerate(dirty_df.columns):
                pred = predicted_cells.get(col_idx, set())
                actual = actual_cells.get(col_idx, set())
                col_TP = len(pred & actual)
                col_FP = len(pred - actual)
                col_FN = len(actual - pred)

                col_precision = col_TP / (col_TP + col_FP) if (col_TP + col_FP) else 0
                col_recall = col_TP / (col_TP + col_FN) if (col_TP + col_FN) else 0
                col_f1 = (2 * col_precision * col_recall / (col_precision + col_recall)) if (col_precision + col_recall) else 0

                col_msg = (
                    f"Column: {col_name:<20} | "
                    f"Precision: {col_precision:.3f} | "
                    f"Recall: {col_recall:.3f} | "
                    f"F1: {col_f1:.3f}\n"
                )
                print(col_msg.strip())
                f.write(col_msg)

            # Aggregate overall
            overall_TP += TP
            overall_FP += FP
            overall_FN += FN
            overall_TN += TN

        # Moved overall metrics **outside** dataset loop
        overall_precision = overall_TP / (overall_TP + overall_FP) if overall_TP + overall_FP else 0
        overall_recall = overall_TP / (overall_TP + overall_FN) if overall_TP + overall_FN else 0
        overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (
                    overall_precision + overall_recall) else 0

        summary = (
            f"\n===== Overall Metrics across all datasets =====\n"
            f"Configuration: {config}\n"
            f"TP: {overall_TP}, FP: {overall_FP}, FN: {overall_FN}, TN: {overall_TN}\n"
            f"Precision: {overall_precision:.3f}\n"
            f"Recall: {overall_recall:.3f}\n"
            f"F1 Score: {overall_f1:.3f}\n"
        )
        print(summary)
        f.write(summary)

    print(f"\nResults have been saved to {output_file}")

    '''
        print("\n Per-Column Metrics:")
        total_cells = dirty_df.shape[0] * dirty_df.shape[1]
        TN = total_cells - (TP + FP + FN) # cells correctly predicted as not errors

        for col_idx, col_name in enumerate(dirty_df.columns):
            pred = predicted_cells.get(col_idx, set())
            actual = actual_cells.get(col_idx, set())
            TP = len(pred & actual)
            FP = len(pred - actual)
            FN = len(actual - pred)

            precision = TP / (TP + FP) if (TP + FP) else 0
            recall = TP / (TP + FN) if (TP + FN) else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

            print(f"Column: {col_name:<20} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

        # --- Dataset-level metrics ---
        all_predicted = set((dataset_name, col_idx, row_idx) for col_idx, rows in predicted_cells.items() for row_idx in rows)
        all_actual = set((dataset_name, col_idx, row_idx) for col_idx, rows in actual_cells.items() for row_idx in rows)

        TP = len(all_predicted & all_actual)
        FP = len(all_predicted - all_actual)
        FN = len(all_actual - all_predicted)

        dataset_precision = TP / (TP + FP) if TP + FP else 0
        dataset_recall = TP / (TP + FN) if TP + FN else 0
        dataset_f1 = (2 * dataset_precision * dataset_recall / (dataset_precision + dataset_recall)) if (dataset_precision + dataset_recall) else 0

        print(f"\n Dataset Metrics for {dataset_name}: Precision={dataset_precision:.3f}, Recall={dataset_recall:.3f}, F1={dataset_f1:.3f}")

        # Aggregate for overall
        overall_TP += TP # predicted as errors & actual errors
        overall_FP += FP # predicted as errors & not actual errors
        overall_FN += FN # actual errors but not predicted errors
        overall_TN += TN

        # Print actual counts
        print("\n Actual Error Counts (by column):")
        for (table, col), indices in actual_errors_by_column.items():
            if table == dataset_name:
                print(f"Table: {table} | Column: {col} | Actual Error Count: {len(indices)}")

    # --- Overall metrics ---
    overall_precision = overall_TP / (overall_TP + overall_FP) if overall_TP + overall_FP else 0
    overall_recall = overall_TP / (overall_TP + overall_FN) if overall_TP + overall_FN else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) else 0

    print("\n Overall Metrics across all datasets:")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall:    {overall_recall:.3f}")
    print(f"F1 Score:  {overall_f1:.3f}")
    with open(output_file, "w") as f:
        f.write("===== Overall Metrics across all datasets =====\n")
        f.write(f"True Positives (TP): {overall_TP}\n")
        f.write(f"False Positives (FP): {overall_FP}\n")
        f.write(f"False Negatives (FN): {overall_FN}\n")
        f.write(f"True Negatives (TN): {overall_TN}\n")

        overall_precision = overall_TP / (overall_TP + overall_FP) if overall_TP + overall_FP else 0
        overall_recall = overall_TP / (overall_TP + overall_FN) if overall_TP + overall_FN else 0
        overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (
                    overall_precision + overall_recall) else 0

        f.write(f"Precision: {overall_precision:.3f}\n")
        f.write(f"Recall: {overall_recall:.3f}\n")
        f.write(f"F1 Score: {overall_f1:.3f}\n")

    print(f"\n Results have been saved to {output_file}")
'''