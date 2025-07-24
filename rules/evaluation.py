# rules/evaluation.py
import pandas


def detect_rule_violations(rule, cluster_columns, column_profiles):
    violations = []
    for col in column_profiles:
        if col["column_name"] in cluster_columns and not rule.applies(col):
            violations.append({
                "column_name": col["column_name"],
                "rule": rule.name,
                "reason": rule.description
            })
    return violations


def analyze_rule_in_cluster(rule, cluster_columns, column_profiles):
    passed, violations = [], []

    for col in column_profiles:
        if col["column_name"] not in cluster_columns:
            continue

        if rule.applies(col):
            passed.append(col["column_name"])
        else:
            violations.append({
                "column_name": col["column_name"],
                "rule": rule.name,
                "reason": rule.description
            })

    return {
        "rule": rule.name,
        "cluster_columns": cluster_columns,
        "passed": passed,
        "violated": [v["column_name"] for v in violations],
        "violations": violations,
        "tag": (
            "valid_cluster" if len(passed) == len(cluster_columns) else
            "invalid_cluster" if len(passed) == 0 else
            "mixed_cluster"
        )
    }


def get_shared_rules_per_cluster(rules, column_profiles, clusters, threshold=0.7):
    shared_rules = {}
    # Build a lookup: column_name -> profile
    col_lookup = {col['column_name']: col for col in column_profiles}
    for cid, colnames in clusters.items():
        applicable_rules = []

        for rule in rules:
            applicable = 0
            for colname in colnames:
                col = col_lookup.get(colname)
                if col and rule.applies(col):
                    applicable += 1
            ratio = applicable / len(colnames)
            if ratio >= threshold:
                applicable_rules.append(rule.name)

        shared_rules[cid] = applicable_rules

    return shared_rules


def get_shared_rules_per_cluster_with_sample_cloumn(rules, column_profiles, clusters):
    shared_rules = {}
    col_lookup = {col['column_name']: col for col in column_profiles}

    for cid, colnames in clusters.items():
        applicable_rules = []

        for rule in rules:
            sample_column = getattr(rule, "sample_column", None)

            #print(sample_column)
            if not sample_column or sample_column not in colnames:
                continue
            applicable_rules.append(rule.name)
            #print("test")
            #print(rule.name)

        shared_rules[cid] = applicable_rules

    return shared_rules


def detect_cell_errors_in_clusters(clusters, shared_rules, rules, raw_dataset):
    """
    Apply shared rules to each cell in each column of each cluster.

    Returns:
        List of error cell dicts: {column, row_index, value, rule}
    """
    rule_map = {rule.name: rule for rule in rules}
    errors = []

    for cid, colnames in clusters.items():
        for rule_name in shared_rules.get(cid, []):
            rule = rule_map.get(rule_name)
            if not rule or not hasattr(rule, "validate_cell"):
                continue  # skip rules that don't support cell-level validation

            for col in colnames:
                values = raw_dataset.get(col, [])
                if hasattr(rule, "prepare"):
                    rule.prepare(values)

                for i, val in enumerate(values):
                    if not rule.validate_cell(val):
                        errors.append({
                            "column": col,
                            "row_index": i,
                            "value": val,
                            "rule": rule_name,
                            "cluster": cid
                        })

    return errors

def detect_error_cells(clusters, shared_rules, rules, column_data_lookup, table_name):
    all_errors = []
    rule_lookup = {rule.description: rule for rule in rules}

    for cid, colnames in clusters.items():
        applicable_rule_descriptions = shared_rules.get(cid, [])

        for rule_desc in applicable_rule_descriptions:
            rule = rule_lookup.get(rule_desc)
            print(rule)
            if not rule:
                continue

            for colname in colnames:
                if colname not in column_data_lookup:
                    continue

                col_data = column_data_lookup[colname]
                rule.prepare(col_data)  # if needed

                error_indices = []
                for idx, val in col_data.items():
                    if not rule.validate_cell(val):
                        error_indices.append(idx)

                if error_indices:
                    all_errors.append({
                        "table": table_name,
                        "cluster": cid,
                        "column": colname,
                        "rule": rule_desc,
                        "error_indices": error_indices
                    })

    return all_errors


def detect_error_cells_across_tables(clusters, shared_rules, rules, raw_dataset):
    errors = []
    rule_lookup = {rule.description: rule for rule in rules}

    for cid, colnames in clusters.items():
        applicable_rule_descriptions = shared_rules.get(cid, [])

        for rule_desc in applicable_rule_descriptions:
            rule = rule_lookup.get(rule_desc)
            if not rule:
                continue

            for table_column in colnames:
                # Split into table + column name
                if "_" not in table_column:
                    continue  # skip malformed names
                table_name, column_name = table_column.split("_", 1)

                df = raw_dataset.get(table_name)
                if df is None or column_name not in df.columns:
                    continue  # skip if data not found

                column_data = df[column_name]

                rule.prepare(column_data)  # optional if applicable

                error_indices = [
                    idx for idx, val in column_data.items()
                    if not rule.validate_cell(val)
                ]

                if error_indices:
                    errors.append({
                        "table": table_name,
                        "cluster": cid,
                        "column": column_name,
                        "rule": rule_desc,
                        "error_indices": error_indices
                    })

    return errors

def statistical_cell_detector(series):
    errors = []
    if series.dtype.kind in "biufc":  # numeric columns
        mean, std = series.mean(), series.std()
        for idx, val in series.items():
            if not pandas.isna(val) and abs(val - mean) > 3 * std:
                errors.append(idx)
    elif series.dtype == 'object':
        freq = series.value_counts(normalize=True)
        low_freq_values = set(freq[freq < 0.01].index)
        for idx, val in series.items():
            if val in low_freq_values:
                errors.append(idx)
    return errors

def detect_combined_errors(clusters, shared_rules, rules, raw_dataset):
    errors = []
    rule_lookup = {rule.name: rule for rule in rules}

    for cid, colnames in clusters.items():
        for colname in colnames:
            table, column = colname.split("_", 1)
            df = raw_dataset.get(table)
            if df is None or column not in df.columns:
                continue
            series = df[column]
            col_errors = set()

            # Rule-based detection
            for rule_name in shared_rules.get(cid, []):
                rule = rule_lookup[rule_name]
                rule.prepare(series)
                col_errors.update(idx for idx, val in series.items() if not rule.validate_cell(val))

            # Statistical detection
            stat_errors = statistical_cell_detector(series)
            col_errors.update(stat_errors)

            if col_errors:
                errors.append({
                    "table": table,
                    "cluster": cid,
                    "column": column,
                    "error_indices": sorted(list(col_errors))
                })
    return errors


