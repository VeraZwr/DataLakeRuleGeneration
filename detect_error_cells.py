# detect_error_cells.py

import os
import csv
import pickle
import pandas as pd
import re

def load_rules(rules_path):
    with open(rules_path, 'rb') as f:
        return pickle.load(f)

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path, dtype=str).fillna("")

def rule_violations(value, rule):
    """Basic rule checks. You can expand this for more specific rules."""
    if rule == "MUST NOT be NULL":
        return value.strip() == ""
    if "FIXED LENGTH FIELD" in rule:
        expected_len = int(re.findall(r'\d+', rule)[0])
        return len(value.strip()) != expected_len
    if rule.startswith("NUMERIC FIELD"):
        return not re.match(r'^[\d.,]+$', value.strip())
    if rule.startswith("STRUCTURED FIELD"):
        return not re.match(r'^[\w-]+$', value.strip())
    if rule.startswith("CATEGORICAL FIELD"):
        # This is context-dependent. Use set or thresholds in real apps
        return False  # Placeholder: needs specific value set
    if rule.startswith("LIKELY UNIQUE FIELD"):
        return False  # Not handled at row level easily
    return False

def detect_errors(df, rules_dict):
    errors = []

    for col in df.columns:
        if col not in rules_dict:
            continue
        for rule in rules_dict[col]:
            for idx, val in df[col].items():
                if rule_violations(val, rule):
                    errors.append({
                        "column_name": col,
                        "row_index": idx,
                        "violated_rule": rule
                    })
    return errors

def save_errors(errors, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['column_name', 'row_index', 'violated_rule'])
        writer.writeheader()
        writer.writerows(errors)

if __name__ == "__main__":
    dataset_name = "beers"
    results_folder = "results"
    dataset_folder = os.path.join(results_folder, dataset_name)

    dataset_path = os.path.join("datasets/Quintet", dataset_name, "dirty.csv")
    rules_path = os.path.join(dataset_folder, "dataset_rules.dictionary")
    output_csv_path = os.path.join(dataset_folder, "error_cells.csv")

    df = load_dataset(dataset_path)
    rules_dict = load_rules(rules_path)

    errors = detect_errors(df, rules_dict)
    save_errors(errors, output_csv_path)

    print(f"✔ Detected {len(errors)} error(s). Results saved to: {output_csv_path}")
