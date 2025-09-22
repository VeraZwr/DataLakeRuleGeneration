from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score
from rules.evaluation import detect_combined_errors, detect_dynamic_errors
from utils.file_io import csv_to_column_dict
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
from utils.read_data import read_csv

from rules.dictionary_rule import SIMPLE_RULE_PROFILES

def prf_from_sets(pred_set, gold_set, empty_empty="one"):
    """
    pred_set, gold_set: sets of (table, column or col_idx, row)
    empty_empty: "one" -> return 1.0 for P/R/F1 when both sides empty,
                 "nan" -> return float('nan') to mark as N/A.

    Returns: precision, recall, f1, TP, FP, FN
    """
    TP = len(pred_set & gold_set)
    FP = len(pred_set - gold_set)
    FN = len(gold_set - pred_set)

    # both sides empty
    if (TP + FP) == 0 and (TP + FN) == 0:
        if empty_empty == "nan":
            return float('nan'), float('nan'), float('nan'), TP, FP, FN
        return 1.0, 1.0, 1.0, TP, FP, FN

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
    return precision, recall, f1, TP, FP, FN

def _rule_description_map():
    """Map rule_name -> description pulled from dictionary_rule."""
    desc = {}
    for name, spec in dict(SIMPLE_RULE_PROFILES).items():
        if isinstance(spec, dict):
            d = spec.get("description")
            if isinstance(d, str) and d.strip():
                desc[name] = d.strip()
    return desc

def evaluated_clean_columns(errors, dataset_name, dirty_df, clean_df):
    """
    Map evaluated dirty columns (from predictions) to CLEAN columns by position.
    Returns a set of clean column names that correspond to clustered/evaluated dirty columns.
    """
    eval_dirty = evaluated_columns_from_errors(errors).get(dataset_name, set())
    dirty_pos = {c: i for i, c in enumerate(dirty_df.columns)}
    eval_clean = set()
    for dcol in eval_dirty:
        i = dirty_pos.get(dcol)
        if i is not None and i < clean_df.shape[1]:
            eval_clean.add(clean_df.columns[i])
    return eval_clean

def evaluated_columns_from_errors(errors):
    # returns {table: set({col_name, ...}), ...}
    cols = defaultdict(set)
    for e in errors:
        if e.get("table") and e.get("column"):
            cols[e["table"]].add(e["column"])
    return cols

def _collect_cell_annotations(errors_raw, default_desc="ERROR"):
    """
    Build (table, column, row) -> description mapping from the *raw* errors list
    returned by detect_combined_errors (BEFORE merge_errors()).

    We try to resolve rule description via SIMPLE_RULE_PROFILES. If unavailable,
    we fall back to any 'description' present in the error item; else default_desc.
    If multiple rules flag the same cell, we join descriptions with ' | '.
    """
    desc_map = _rule_description_map()
    cell_to_descs = {}  # (tbl, col, row) -> set(descriptions)

    for e in errors_raw:
        tbl = e.get("table")
        col = e.get("column")
        idxs = e.get("error_indices") or []
        if not tbl or not col:
            continue

        # choose description
        desc = (e.get("description") or
                SIMPLE_RULE_PROFILES.get(e.get("rule_name"), {}).get("description") or
                default_desc)

        for r in idxs:
            key = (tbl, col, int(r))
            s = cell_to_descs.setdefault(key, set())
            s.add(desc)

        # try to figure out rule key(s)
        cand_keys = [e.get("rule_name"), e.get("rule"), e.get("name"), e.get("rule_id")]
        rule_desc = None
        for rk in cand_keys:
            if isinstance(rk, str) and rk in desc_map:
                rule_desc = desc_map[rk]
                break
        if rule_desc is None:
            # fall back to provided description if any
            if isinstance(e.get("description"), str) and e["description"].strip():
                rule_desc = e["description"].strip()
            else:
                rule_desc = default_desc

        if not tbl or not col:
            continue
        for r in idxs:
            key = (tbl, col, int(r))
            s = cell_to_descs.setdefault(key, set())
            s.add(rule_desc)

    # collapse to single string; if multiple, join
    return {k: " | ".join(sorted(v)) for k, v in cell_to_descs.items()}

def _export_annotation_grid(dataset_group: str, dataset_name: str,
                            method_label: str,
                            dirty_df: pd.DataFrame,
                            cell_annotations: dict,
                            export_root: Path = Path("error_exports")) -> Path:
    """
    Create a DataFrame same shape as dirty_df, fill 'CLN' then write descriptions for flagged cells.
    Returns the written path.
    """
    annot = pd.DataFrame("CLN", index=dirty_df.index, columns=dirty_df.columns)
    # set descriptions on errors
    for (tbl, col, row), desc in cell_annotations.items():
        if col in annot.columns and 0 <= row < len(annot):
            annot.at[row, col] = desc

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = export_root / dataset_group / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"annotations_{dataset_name}_{method_label}_{ts}.csv"
    annot.to_csv(out_path, index=False)
    print(f"[EXPORT] Annotation grid -> {out_path}")
    return out_path

def _flatten_errors_for_export(errors, raw_dataset):
    """
    errors: [{table, column, error_indices:[...]}, ...]
    raw_dataset: {table_name: pd.DataFrame}
    -> list of rows with one row per erroneous cell.
    """
    rows = []
    for err in errors:
        tbl = err["table"]
        col = err["column"]
        df = raw_dataset.get(tbl)
        for idx in err["error_indices"]:
            val = None
            if df is not None and col in df.columns and 0 <= idx < len(df):
                try:
                    val = df.at[idx, col]
                except Exception:
                    val = None
            rows.append({"table": tbl, "column": col, "row_index": idx, "dirty_value": val})
    return rows

def _export_errors_csv(rows, export_path: Path):
    export_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(export_path, index=False)
    print(f"[EXPORT] Wrote {len(rows)} rows -> {export_path}")

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


def compute_actual_errors(clean_dataset_dict, dirty_dataset_dict, columns_to_keep_dirty=None):
    """
    Compare dirty vs clean by POSITION (col 0↔0, 1↔1, …), never by names.
    Records keys as (table_name, DIRTY_COL_NAME) so predictions align.
    If columns_to_keep_dirty is provided {table: set(dirty_col_names)}, ONLY those dirty columns are scored.
    """
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

        if clean_df.shape[0] != dirty_df.shape[0]:
            print(f"Row count mismatch in table '{table_name}', skipping")
            continue

        min_cols = min(clean_df.shape[1], dirty_df.shape[1])
        keep = None
        if columns_to_keep_dirty is not None:
            keep = columns_to_keep_dirty.get(table_name, set())

        for row_idx in range(len(clean_df)):
            for col_idx in range(min_cols):
                try:
                    dirty_col_name = dirty_df.columns[col_idx]  # use DIRTY name for key
                    if keep is not None and dirty_col_name not in keep:
                        continue

                    clean_val = str(clean_df.iat[row_idx, col_idx])   # position-based
                    dirty_val = str(dirty_df.iat[row_idx, col_idx])   # position-based
                    if clean_val != dirty_val:
                        actual_errors_by_column[(table_name, dirty_col_name)].append(row_idx)
                except Exception as e:
                    print(f"Error comparing cell [{row_idx}, {dirty_col_name}] in table '{table_name}': {e}")

    return actual_errors_by_column


def evaluate_one_dataset_only(rules, shared_rules, clusters, column_profiles,
                              dataset_group, dataset_name, config,
                              method_label="DBSCAN",
                              export_root: Path = Path("error_exports")):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = Path("datasets") / dataset_group / dataset_name
    dirty_df = read_csv(dataset_path / "dirty.csv")
    clean_df = read_csv(dataset_path / "clean.csv")

    raw_dataset = {dataset_name: dirty_df}
    clean_dataset_dict = {dataset_name: clean_df}

    print(f"\n Detecting errors for {dataset_name}...")
    errors_raw = detect_combined_errors(clusters, shared_rules, rules, raw_dataset, column_profiles)

    # build & export annotation grid (CLN / description) using *raw* errors
    cell_ann = _collect_cell_annotations(errors_raw, default_desc="ERROR")
    _export_annotation_grid(dataset_group, dataset_name, method_label, dirty_df, cell_ann, export_root=export_root)

    errors = merge_errors(errors_raw)

    # Print detected errors
    for err in errors:
        if err['table'] == dataset_name:
            err_count = len(err['error_indices'])
            err_values = [dirty_df.at[idx, err['column']] for idx in err['error_indices']]
            print(f"Table: {err['table']} | Column: {err['column']} | Error count: {err_count}")
            # print(f"Error rows: {err['error_indices']}")
            # print(f"Error values: {err_values}\n")

    # --- Compute TP, FP, FN, TN ---
    evaluated_cols_dirty = evaluated_columns_from_errors(errors)

    # predicted (table, dirty_col_name, row)
    predicted = set(
        (dataset_name, err["column"], row)
        for err in errors if err["table"] == dataset_name
        for row in err["error_indices"]
    )

    # ACTUAL restricted to evaluated (DIRTY) columns
    actual_by_column = compute_actual_errors(
        clean_dataset_dict, raw_dataset,
        columns_to_keep_dirty=evaluated_cols_dirty
    )
    actual = set(
        (tbl, col, row)
        for (tbl, col), rows in actual_by_column.items()
        if tbl == dataset_name
        for row in rows
    )

    # safe PRF
    precision, recall, f1, TP, FP, FN = prf_from_sets(predicted, actual, empty_empty="one")

    # TN over ONLY scored cells (rows × evaluated dirty columns)
    scored_cols = len(evaluated_cols_dirty.get(dataset_name, set()))
    scored_cells = scored_cols * dirty_df.shape[0]
    TN = scored_cells - (TP + FP + FN)

    # --- Metrics ---
    # precision = TP / (TP + FP) if TP + FP else 0
    # recall = TP / (TP + FN) if TP + FN else 0
    # f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f" Evaluation Metrics for {dataset_name}:")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    # --- Per-Column Metrics (position-aligned; unclustered CLEAN columns => 1.0) ---
    print("\n Per-Column Metrics:")

    # Columns actually evaluated (DIRTY names) for this dataset
    evaluated_dirty_cols = evaluated_columns_from_errors(errors).get(dataset_name, set())

    min_cols = min(dirty_df.shape[1], clean_df.shape[1])

    # fast index of predictions per dirty column
    preds_by_col = defaultdict(set)
    for (tbl, col, r) in predicted:
        if tbl == dataset_name:
            preds_by_col[col].add(r)

    for col_idx in range(min_cols):
        dirty_col = dirty_df.columns[col_idx]
        clean_col = clean_df.columns[col_idx]

        if dirty_col not in evaluated_dirty_cols:
            # Not included in this method's clustering/sampling -> treat as perfect per your policy
            col_precision = col_recall = col_f1 = 1.0
        else:
            pred_rows = preds_by_col.get(dirty_col, set())
            gold_rows = set(actual_by_column.get((dataset_name, dirty_col), []))

            # robust: empty-empty => 1/1/1
            p, r, f, *_ = prf_from_sets(
                {(dataset_name, dirty_col, rr) for rr in pred_rows},
                {(dataset_name, dirty_col, rr) for rr in gold_rows},
                empty_empty="one"
            )
            col_precision, col_recall, col_f1 = p, r, f

        print(
            f"Column: {dirty_col:<20} (clean: {clean_col}) | "
            f"P: {col_precision:.3f} | R: {col_recall:.3f} | F1: {col_f1:.3f}"
        )
        #print(col_msg.strip())

    # --- Actual error counts ---
    # actual_errors_by_column = compute_actual_errors(clean_dataset_dict, raw_dataset)
    actual_errors_by_column = actual_by_column
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
    output_file = Path("output") /dataset_group/ f"evaluation_results_{dataset_name}_{method_label}_{timestamp}.txt"
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
    # Export erroneous cells to CSV (one row per cell)
    export_rows = _flatten_errors_for_export(errors, raw_dataset)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_csv = export_root / dataset_group / dataset_name / f"errors_{dataset_name}_{method_label}_{ts}.csv"
    _export_errors_csv(export_rows, export_csv)



def evaluate_multiple_datasets(
    rules, shared_rules, clusters, column_profiles,
    dataset_group, config,
    method_label="DBSCAN",
    export_root: Path = Path("error_exports")
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_TP = overall_FP = overall_FN = overall_TN = 0
    output_dir = Path("output") / dataset_group
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"evaluation_results_{dataset_group}_{method_label}_{timestamp}.txt"

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
                print(msg.strip()); f.write(msg)
                continue

            dirty_df = read_csv(dirty_file)
            clean_df = read_csv(clean_file)
            raw_dataset = {dataset_name: dirty_df}
            clean_dataset_dict = {dataset_name: clean_df}

            # Detect
            errors_raw = detect_combined_errors(clusters, shared_rules, rules, raw_dataset, column_profiles)
            cell_ann = _collect_cell_annotations(errors_raw, default_desc="ERROR")
            _export_annotation_grid(dataset_group, dataset_name, method_label, dirty_df, cell_ann, export_root=export_root)
            errors = merge_errors(errors_raw)

            # Evaluate only the DIRTY columns you actually predicted on (clustered)
            evaluated_cols_dirty = evaluated_columns_from_errors(errors)

            # Predicted set (table, dirty_col, row)
            predicted = set(
                (dataset_name, err["column"], row)
                for err in errors if err["table"] == dataset_name
                for row in err["error_indices"]
            )

            # Actual differences by POSITION, restricted to evaluated DIRTY columns
            actual_by_column = compute_actual_errors(
                clean_dataset_dict, raw_dataset,
                columns_to_keep_dirty=evaluated_cols_dirty
            )
            actual = set(
                (tbl, col, row)
                for (tbl, col), rows in actual_by_column.items()
                if tbl == dataset_name
                for row in rows
            )

            # Micro metrics
            precision, recall, f1, TP, FP, FN = prf_from_sets(predicted, actual, empty_empty="one")

            # TN over ONLY scored cells (rows × evaluated dirty columns)
            scored_cols = len(evaluated_cols_dirty.get(dataset_name, set()))
            scored_cells = scored_cols * dirty_df.shape[0]
            TN = max(0, scored_cells - (TP + FP + FN))

            msg = (
                f"\nDataset Metrics for {dataset_name}:\n"
                f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}\n"
                f"Precision: {precision:.3f}\n"
                f"Recall: {recall:.3f}\n"
                f"F1 Score: {f1:.3f}\n"
                f"--- Per-Column Metrics ---\n"
            )
            print(msg); f.write(msg)

            # --- Per-Column Metrics (position-aligned; unclustered CLEAN columns => 1.0) ---
            eval_dirty = evaluated_columns_from_errors(errors).get(dataset_name, set())
            dirty_pos = {c: i for i, c in enumerate(dirty_df.columns)}
            eval_clean = set(clean_df.columns[i] for c in eval_dirty if (i := dirty_pos.get(c)) is not None and i < clean_df.shape[1])

            min_cols = min(dirty_df.shape[1], clean_df.shape[1])
            for col_idx in range(min_cols):
                dirty_col = dirty_df.columns[col_idx]
                clean_col = clean_df.columns[col_idx]

                if clean_col not in eval_clean:
                    col_precision = col_recall = col_f1 = 1.0
                else:
                    pred_rows = {r for (tbl, col, r) in predicted if tbl == dataset_name and col == dirty_col}
                    gold_rows = set(actual_by_column.get((dataset_name, dirty_col), []))
                    p, r, ff, *_ = prf_from_sets(
                        {(dataset_name, dirty_col, rr) for rr in pred_rows},
                        {(dataset_name, dirty_col, rr) for rr in gold_rows},
                        empty_empty="one"
                    )
                    col_precision, col_recall, col_f1 = p, r, ff

                col_msg = (
                    f"Column: {dirty_col:<20} (clean: {clean_col}) | "
                    f"Precision: {col_precision:.3f} | Recall: {col_recall:.3f} | F1: {col_f1:.3f}\n"
                )
                print(col_msg.strip()); f.write(col_msg)

            # aggregate micro across datasets
            overall_TP += TP; overall_FP += FP; overall_FN += FN; overall_TN += TN

            # Export erroneous cells for this dataset
            # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # export_rows = _flatten_errors_for_export(errors, raw_dataset)
            # export_csv = export_root / dataset_group / dataset_name / f"errors_{dataset_name}_{method_label}_{ts}.csv"
            # _export_errors_csv(export_rows, export_csv)

        # Overall micro
        overall_precision = overall_TP / (overall_TP + overall_FP) if (overall_TP + overall_FP) else 0
        overall_recall    = overall_TP / (overall_TP + overall_FN) if (overall_TP + overall_FN) else 0
        overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) else 0

        summary = (
            f"\n===== Overall Metrics across all datasets =====\n"
            f"Configuration: {config}\n"
            f"TP: {overall_TP}, FP: {overall_FP}, FN: {overall_FN}, TN: {overall_TN}\n"
            f"Precision: {overall_precision:.3f}\n"
            f"Recall: {overall_recall:.3f}\n"
            f"F1 Score: {overall_f1:.3f}\n"
        )
        print(summary); f.write(summary)

    print(f"\nResults have been saved to {output_file}")


    # Export erroneous cells for this dataset
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_rows = _flatten_errors_for_export(errors, raw_dataset)
    export_csv = export_root / dataset_group / dataset_name / f"errors_{dataset_name}_{method_label}_{ts}.csv"
    _export_errors_csv(export_rows, export_csv)

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